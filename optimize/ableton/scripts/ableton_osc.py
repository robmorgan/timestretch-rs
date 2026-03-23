#!/usr/bin/env python3
"""
Control Ableton Live via AbletonOSC.

Provides both a library API (for use by render_via_ableton.py) and a CLI.

Requires AbletonOSC MIDI Remote Script installed and enabled in Ableton.
https://github.com/ideoforms/AbletonOSC

Usage:
    python ableton_osc.py set-tempo 115
    python ableton_osc.py get-tempo
    python ableton_osc.py set-loop off
    python ableton_osc.py configure 115.0 --no-loop
"""
import argparse
import sys
import time
import threading

from pythonosc import udp_client, osc_server, dispatcher


ABLETON_OSC_HOST = "127.0.0.1"
ABLETON_OSC_SEND_PORT = 11000   # AbletonOSC listens here
ABLETON_OSC_RECV_PORT = 11001   # AbletonOSC sends replies here


# ---------------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------------

def send_message(address, *args):
    """Send an OSC message to AbletonOSC (fire-and-forget)."""
    client = udp_client.SimpleUDPClient(ABLETON_OSC_HOST, ABLETON_OSC_SEND_PORT)
    client.send_message(address, list(args))


def send_and_receive(address, *args, timeout=2.0):
    """Send an OSC message and wait for a reply."""
    result = {"value": None}

    def handler(addr, *reply_args):
        result["value"] = reply_args

    disp = dispatcher.Dispatcher()
    disp.set_default_handler(handler)

    server = osc_server.ThreadingOSCUDPServer(
        (ABLETON_OSC_HOST, ABLETON_OSC_RECV_PORT), disp
    )
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    client = udp_client.SimpleUDPClient(ABLETON_OSC_HOST, ABLETON_OSC_SEND_PORT)
    client.send_message(address, list(args))

    deadline = time.time() + timeout
    while result["value"] is None and time.time() < deadline:
        time.sleep(0.05)

    server.shutdown()
    return result["value"]


def is_osc_available(timeout=2.0):
    """Check whether AbletonOSC is reachable."""
    result = send_and_receive("/live/song/get/tempo", timeout=timeout)
    return result is not None


def set_tempo(bpm):
    """Set the master tempo. Returns the confirmed tempo or None."""
    send_message("/live/song/set/tempo", float(bpm))
    time.sleep(0.2)
    result = send_and_receive("/live/song/get/tempo", timeout=2.0)
    if result:
        return result[0]
    return None


def get_tempo():
    """Get the current master tempo. Returns float or None."""
    result = send_and_receive("/live/song/get/tempo", timeout=2.0)
    if result:
        return result[0]
    return None


def stop_playback():
    """Stop playback if currently playing."""
    send_message("/live/song/stop_playing")


def get_clip_length(track_idx=0, clip_idx=0):
    """Get clip length in beats. Returns float or None."""
    result = send_and_receive("/live/clip/get/length", track_idx, clip_idx,
                              timeout=2.0)
    if result:
        return result[0]
    return None


def set_song_loop(start_beats, length_beats):
    """Set the arrangement loop range (used as export range)."""
    send_message("/live/song/set/loop_start", float(start_beats))
    time.sleep(0.1)
    send_message("/live/song/set/loop_length", float(length_beats))


def set_clip_loop(track_idx, clip_idx, enabled):
    """Enable or disable looping on a clip."""
    send_message("/live/clip/set/looping", track_idx, clip_idx,
                 1 if enabled else 0)


def configure_session(bpm, disable_loop=True, track_idx=0, clip_idx=0):
    """
    Configure an Ableton session for rendering: set tempo, disable loop.

    Returns (success: bool, details: str).
    """
    issues = []

    # Stop playback to avoid "stop audio" dialog during export
    stop_playback()

    # Set tempo
    confirmed = set_tempo(bpm)
    if confirmed is not None:
        if abs(confirmed - bpm) > 0.1:
            issues.append(f"Tempo mismatch: requested {bpm}, got {confirmed}")
    else:
        issues.append("Could not verify tempo (no OSC reply)")

    # Disable clip loop
    if disable_loop:
        set_clip_loop(track_idx, clip_idx, enabled=False)

    if issues:
        return False, "; ".join(issues)
    return True, f"Tempo={confirmed} BPM, loop={'off' if disable_loop else 'on'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Control Ableton via AbletonOSC")
    subparsers = parser.add_subparsers(dest="command")

    # set-tempo
    tempo_parser = subparsers.add_parser("set-tempo", help="Set master tempo")
    tempo_parser.add_argument("bpm", type=float, help="Target BPM")

    # get-tempo
    subparsers.add_parser("get-tempo", help="Get current master tempo")

    # set-loop
    loop_parser = subparsers.add_parser("set-loop", help="Set clip loop on/off")
    loop_parser.add_argument("state", choices=["on", "off"], help="Loop state")
    loop_parser.add_argument("--track", type=int, default=0, help="Track index")
    loop_parser.add_argument("--clip", type=int, default=0, help="Clip index")

    # configure (combined: tempo + loop)
    conf_parser = subparsers.add_parser("configure",
                                        help="Set tempo and disable loop")
    conf_parser.add_argument("bpm", type=float, help="Target BPM")
    conf_parser.add_argument("--no-loop", action="store_true", default=True,
                             help="Disable clip loop (default)")
    conf_parser.add_argument("--loop", action="store_true",
                             help="Keep clip loop enabled")

    args = parser.parse_args()

    if args.command == "set-tempo":
        confirmed = set_tempo(args.bpm)
        if confirmed is not None:
            print(f"Tempo: {confirmed} BPM")
        else:
            print(f"Sent tempo={args.bpm}. Could not verify (no OSC reply).")

    elif args.command == "get-tempo":
        tempo = get_tempo()
        if tempo is not None:
            print(f"Current tempo: {tempo} BPM")
        else:
            print("No response. Is AbletonOSC enabled?", file=sys.stderr)
            sys.exit(1)

    elif args.command == "set-loop":
        enabled = args.state == "on"
        set_clip_loop(args.track, args.clip, enabled)
        print(f"Loop {'on' if enabled else 'off'} (track={args.track}, clip={args.clip})")

    elif args.command == "configure":
        disable_loop = not args.loop
        ok, details = configure_session(args.bpm, disable_loop=disable_loop)
        print(f"{'OK' if ok else 'WARN'}: {details}")
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
