-- Headless Elastique reference renders through REAPER.
--
-- REAPER licenses zplane's élastique and ships it as the "élastique 3.3.3
-- Pro" pitch-shift mode — the same engine as Elastique Pro V3 and Ableton's
-- Complex Pro. This script renders a job list with that mode so the Parity
-- Track's Elastique arm (ROADMAP.md, Stage 23) is reproducible from the
-- shell instead of hand-exported.
--
-- Driven by scripts/render_elastique.py, which writes the job file and
-- launches:
--   REAPER -nosplash -ignoreerrors scripts/reaper_elastique.lua
-- The job file path comes from TIMESTRETCH_REAPER_JOBS (a Lua chunk
-- returning a table); the same directory receives `<jobs>.log` and
-- `<jobs>.done` so the wrapper can tell success from a hung launch.
--
-- Job fields: src (absolute path), out_dir, out_name (no extension),
-- rate (tempo rate: 1.08 = 8 % faster; output length = source / rate).
-- Each job renders the whole source at the source sample rate and channel
-- count, 32-bit float WAV, no dither, no tail.

local jobs_path = os.getenv("TIMESTRETCH_REAPER_JOBS")
if not jobs_path then
  reaper.ShowConsoleMsg("TIMESTRETCH_REAPER_JOBS not set\n")
  return
end
local log_path = jobs_path .. ".log"
local log = io.open(log_path, "w")
local function say(msg)
  log:write(msg, "\n")
  log:flush()
end

local ok, jobs = pcall(dofile, jobs_path)
if not ok then
  say("job file failed: " .. tostring(jobs))
  log:close()
  reaper.Main_OnCommand(40004, 0) -- File: Quit REAPER
  return
end

-- Pitch mode by name so this does not depend on REAPER's numbering.
local function find_pitch_mode()
  local i = 0
  while true do
    local ok2, name = reaper.EnumPitchShiftModes(i)
    if not ok2 then break end
    if name and name:lower():find("lastique") and name:find("3%.3%.3") and name:find("Pro") then
      -- submode 0 is "Normal" for the Pro engine.
      return (i << 16) | 0, name
    end
    i = i + 1
  end
  return nil, nil
end

local pitch_mode, pitch_name = find_pitch_mode()
if not pitch_mode then
  say("élastique 3.3.3 Pro pitch mode not found")
  log:close()
  reaper.Main_OnCommand(40004, 0)
  return
end
say("pitch mode: " .. pitch_name .. " (" .. tostring(pitch_mode) .. ")")

-- WAV, 32-bit float: "evaw" + bit depth byte + three zero bytes, base64.
local WAV_32F = "ZXZhdyAAAAA="

local proj = 0
local scratch = jobs_path .. ".RPP"
reaper.Main_SaveProjectEx(proj, scratch, 0)

local function render_job(job)
  local source = reaper.PCM_Source_CreateFromFile(job.src)
  if not source then
    say("FAIL " .. job.out_name .. ": cannot open " .. job.src)
    return false
  end
  local src_len = reaper.GetMediaSourceLength(source)
  local sr = reaper.GetMediaSourceSampleRate(source)
  local ch = reaper.GetMediaSourceNumChannels(source)

  reaper.InsertTrackAtIndex(0, false)
  local track = reaper.GetTrack(proj, 0)
  local item = reaper.AddMediaItemToTrack(track)
  local take = reaper.AddTakeToMediaItem(item)
  reaper.SetMediaItemTake_Source(take, source)
  local out_len = src_len / job.rate
  reaper.SetMediaItemInfo_Value(item, "D_POSITION", 0)
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH", out_len)
  reaper.SetMediaItemInfo_Value(item, "D_FADEINLEN", 0)
  reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", 0)
  reaper.SetMediaItemTakeInfo_Value(take, "D_PLAYRATE", job.rate)
  reaper.SetMediaItemTakeInfo_Value(take, "B_PPITCH", 1)
  reaper.SetMediaItemTakeInfo_Value(take, "I_PITCHMODE", pitch_mode)
  reaper.SetMediaItemTakeInfo_Value(take, "D_PITCH", 0)

  reaper.GetSetProjectInfo(proj, "PROJECT_SRATE", sr, true)
  reaper.GetSetProjectInfo(proj, "PROJECT_SRATE_USE", 1, true)
  reaper.GetSetProjectInfo(proj, "RENDER_SRATE", sr, true)
  reaper.GetSetProjectInfo(proj, "RENDER_CHANNELS", ch, true)
  reaper.GetSetProjectInfo(proj, "RENDER_BOUNDSFLAG", 0, true) -- custom
  reaper.GetSetProjectInfo(proj, "RENDER_STARTPOS", 0, true)
  reaper.GetSetProjectInfo(proj, "RENDER_ENDPOS", out_len, true)
  reaper.GetSetProjectInfo(proj, "RENDER_TAILFLAG", 0, true)
  reaper.GetSetProjectInfo(proj, "RENDER_TAILMS", 0, true)
  reaper.GetSetProjectInfo(proj, "RENDER_SETTINGS", 0, true) -- master mix
  reaper.GetSetProjectInfo(proj, "RENDER_DITHER", 0, true)
  reaper.GetSetProjectInfo(proj, "RENDER_ADDTOPROJ", 0, true)
  reaper.GetSetProjectInfo_String(proj, "RENDER_FORMAT", WAV_32F, true)
  reaper.GetSetProjectInfo_String(proj, "RENDER_FILE", job.out_dir, true)
  reaper.GetSetProjectInfo_String(proj, "RENDER_PATTERN", job.out_name, true)
  reaper.UpdateArrange()

  -- File: Render project, using the most recent render settings,
  -- auto-close render dialog when finished. Blocks until done.
  reaper.Main_OnCommand(42230, 0)

  reaper.DeleteTrack(track)
  local out = job.out_dir .. "/" .. job.out_name .. ".wav"
  local f = io.open(out, "rb")
  if f then
    f:close()
    say(string.format("OK %s rate=%.4f len=%.3fs sr=%d ch=%d", out, job.rate, out_len, sr, ch))
    return true
  end
  say("FAIL " .. job.out_name .. ": no output at " .. out)
  return false
end

local n_ok, n_fail = 0, 0
for _, job in ipairs(jobs) do
  reaper.RecursiveCreateDirectory(job.out_dir, 0)
  if render_job(job) then n_ok = n_ok + 1 else n_fail = n_fail + 1 end
end
say(string.format("done: %d ok, %d failed", n_ok, n_fail))
log:close()

local done = io.open(jobs_path .. ".done", "w")
done:write(string.format("%d %d\n", n_ok, n_fail))
done:close()

-- Leave the scratch project clean so quitting does not prompt.
reaper.Main_SaveProjectEx(proj, scratch, 0)
reaper.Main_OnCommand(40004, 0)
