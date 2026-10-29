use super::phase_vocoder::PhaseVocoder;
impl PhaseVocoder {
    fn rewrite_states(&mut self, from: i32, to: i32) {
        for r in 0..self.selector.count {
            let k = self.selector.references[r];
            if self.current_state[k] == from {
                self.current_state[k] = to;
            }
        }
    }
    pub(super) fn classify(&mut self) {
        let eps = 1e-24f32;
        let low = (self.n >> 8).max(1);
        let age =
            (self.state.frame_counter.max(3) - 1).min(((self.n / self.state.input_hop) >> 2) + 1);
        let older = &self.history[age];
        self.current_state[0] = 0;
        self.change[0] = 0.0;
        self.location[0] = 0.0;
        let mut low_energy = 0.0;
        let mut other_energy = eps;
        let mut global_cross = 0.0;
        if low > 1 {
            let mut prior = eps;
            let mut energy = eps;
            let mut negative = 0.0;
            for (k, &old) in older.iter().enumerate().take(low).skip(1) {
                let now = self.magnitude[0][k];
                let delta = (now - old) * (now - old);
                if delta > 1e-14 {
                    self.change[0] += delta;
                }
                prior = old.mul_add(old, prior);
                energy = now.mul_add(now, energy);
                negative -= self.time_cross[k];
                if k <= low >> 1 {
                    low_energy = now.mul_add(now, low_energy);
                } else {
                    other_energy = now.mul_add(now, other_energy);
                    global_cross -= self.time_cross[k];
                }
            }
            self.change[0] /= prior;
            self.location[0] = negative / energy;
            for r in 0..low - 1 {
                self.change[r] = self.change[0];
                self.location[r] = self.location[0];
                self.energy[r] = low_energy / (low - 1) as f32;
            }
        } else {
            self.change[0] /= eps;
        }
        let mut bin = low - 2;
        for region in low - 2..self.selector.count {
            let reference = self.selector.references[region];
            let mut prior = eps;
            let mut energy = eps;
            let mut negative = 0.0;
            self.change[region] = 0.0;
            self.location[reference] = 0.0;
            while bin < self.n / 2 && self.selector.map[bin] <= reference {
                let now = self.magnitude[0][bin];
                let old = older[bin];
                let delta = (now - old) * (now - old);
                if delta > 1e-14 {
                    self.change[region] += delta;
                }
                prior = old.mul_add(old, prior);
                energy = now.mul_add(now, energy);
                negative -= self.time_cross[bin];
                bin += 1;
            }
            self.energy[region] = energy;
            other_energy += energy;
            self.change[region] /= prior;
            global_cross += negative;
            self.location[reference] = negative / energy;
        }
        let alpha = (self.state.input_hop as f32 / (self.n >> 3) as f32).min(1.0);
        let global_location = global_cross / (other_energy + eps);
        let local_low = if self.infinite { -0.5 } else { -0.2 };
        let local_high = if self.infinite { 0.5 } else { 0.1 };
        let global_low = if self.infinite {
            -0.5
        } else if self.state.target_stretch >= 1.5 {
            -0.25
        } else {
            -0.225
        };
        let mut onset_energy = 0.0;
        let mut hold_energy = 0.0;
        let mut compression_energy = 0.0;
        let mut found = false;
        for r in 0..self.selector.count {
            let reference = self.selector.references[r];
            let previous = self.saved_state[reference];
            let change = self.change[r];
            let location = self.location[reference];
            self.current_state[reference] = if change <= 0.75 {
                0
            } else if location <= alpha * local_low || global_location <= global_low {
                hold_energy += self.energy[r];
                if self.state.frame_counter == 1 { 2 } else { 1 }
            } else if location < alpha * local_high || previous == 1 {
                let history_allows = if change > 5.0 {
                    previous != 2
                } else {
                    previous == 1
                };
                if history_allows && self.state.frame_counter > self.state.input_hop / (self.n >> 2)
                {
                    onset_energy += self.energy[r];
                    found = true;
                    2
                } else {
                    0
                }
            } else if self.state.target_stretch != self.state.current_stretch && previous == 2 {
                hold_energy += self.energy[r];
                1
            } else if self.state.current_stretch < 0.5 {
                compression_energy += self.energy[r];
                3
            } else if self.state.frame_counter == 1 {
                2
            } else {
                0
            };
        }
        self.state.alignment_offset = 0;
        if found {
            let mut maximum = f32::NEG_INFINITY;
            let mut index = 0;
            for (i, value) in self.centered.iter_mut().enumerate() {
                *value = value.abs();
                if *value > maximum {
                    maximum = *value;
                    index = i;
                }
            }
            self.state.alignment_offset = if index > self.state.input_hop {
                if index > self.n - self.state.input_hop {
                    index as i32 - self.n as i32
                } else {
                    0
                }
            } else {
                index as i32
            };
        }
        let half = (self.n / 2) as i32;
        let half_hop = (self.state.input_hop >> 1) as i32;
        if self.state.transient_counter <= half
            && self.state.transient_counter >= self.state.transient_target - half_hop
            && self.sync_enabled
        {
            if self.state.transient_counter <= self.state.transient_target + half_hop {
                self.state.alignment_offset = self.state.transient_target;
                self.rewrite_states(1, 2);
            } else {
                self.rewrite_states(2, 1);
                if self.state.target_stretch == 1.0 {
                    self.state.original_spectrum_bypass = true;
                }
            }
            return;
        }
        let total = low_energy + other_energy;
        let mut bypass = false;
        if onset_energy / total > 0.2 {
            self.rewrite_states(1, 2);
            bypass = self.state.target_stretch == 1.0 && self.state.transient_counter > -half;
        }
        if hold_energy / total > 0.2
            && global_location < alpha * local_low
            && hold_energy > onset_energy
        {
            self.rewrite_states(2, 1);
            return;
        }
        if compression_energy / total > 0.4 {
            self.rewrite_states(0, 3);
        }
        if bypass {
            self.state.original_spectrum_bypass = true;
        }
    }
}
