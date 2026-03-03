#!/usr/bin/env python3
"""
Generate Ableton Live .als project files from a template.

Manipulates the ALS XML to set tempo, warp mode (Complex Pro), and insert
an audio clip reference. The generated .als can be opened in Ableton for
rendering.

IMPORTANT: We use regex-based text manipulation instead of xml.etree.ElementTree
for the create/modify path because ET silently drops sibling elements with the
same tag (e.g. multiple <AudioTrack> nodes), corrupting the project. ET is only
used for read-only inspection where element loss doesn't matter.

Usage:
    python create_als_project.py create template.als input.wav 115.0 output.als
    python create_als_project.py inspect template.als
    python create_als_project.py decompress template.als template.xml
"""
import argparse
import gzip
import os
import re
import sys
import xml.etree.ElementTree as ET


# Ableton warp modes:
#   0 = Beats, 1 = Tones, 2 = Texture, 3 = Re-Pitch, 4 = Complex, 6 = Complex Pro
WARP_MODE_COMPLEX_PRO = 6


# --- Raw XML text manipulation (preserves full document structure) ---

def read_als_raw(als_path):
    """Read an .als file and return the raw XML string."""
    with gzip.open(als_path, "rb") as f:
        return f.read().decode("utf-8")


def write_als_raw(xml_text, als_path):
    """Write raw XML text to a gzip-compressed .als file."""
    with gzip.open(als_path, "wb") as f:
        f.write(xml_text.encode("utf-8"))


def set_tempo_raw(xml, bpm):
    """Set all tempo values in the raw XML string."""
    # Ableton stores tempo as integer if whole, float otherwise
    bpm_val = float(bpm)
    bpm_str = str(int(bpm_val)) if bpm_val == int(bpm_val) else str(bpm_val)

    # Master tempo: <Manual Value="120" /> inside <Tempo> block
    # We need to target Manual elements that are children of Tempo.
    # Match the Tempo block and replace its Manual Value.
    def replace_tempo_manual(m):
        return re.sub(
            r'(<Manual\s+Value=")[^"]*(")',
            rf'\g<1>{bpm_str}\2',
            m.group(0),
            count=1,
        )
    xml = re.sub(r'<Tempo>\s*.*?</Tempo>', replace_tempo_manual, xml, flags=re.DOTALL)

    # Per-scene tempo: <Tempo Value="120" />
    xml = re.sub(
        r'(<Tempo\s+Value=")[^"]*(")',
        rf'\g<1>{bpm_str}\2',
        xml,
    )

    return xml


def set_warp_mode_raw(xml, mode=WARP_MODE_COMPLEX_PRO):
    """Set all WarpMode values."""
    return re.sub(
        r'(<WarpMode\s+Value=")[^"]*(")',
        rf'\g<1>{mode}\2',
        xml,
    )


def set_warping_enabled_raw(xml, enabled=True):
    """Enable or disable warping on all clips."""
    value = "true" if enabled else "false"
    xml = re.sub(r'(<IsWarped\s+Value=")[^"]*(")', rf'\g<1>{value}\2', xml)
    xml = re.sub(r'(<Warped\s+Value=")[^"]*(")', rf'\g<1>{value}\2', xml)
    return xml


def set_warp_markers_raw(xml, source_bpm, wav_duration):
    """Rewrite warp markers to reflect the source file's actual BPM.

    Ableton uses warp markers to define the tempo mapping between real time
    (SecTime) and musical time (BeatTime). Two markers are needed:
      - Marker 0 at the origin (0, 0)
      - Marker 1 at the end of the file
    """
    total_beats = wav_duration * source_bpm / 60.0

    new_markers = (
        '<WarpMarkers>\n'
        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'
        '<WarpMarker Id="0" SecTime="0" BeatTime="0" />\n'
        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'
        f'<WarpMarker Id="1" SecTime="{wav_duration}" '
        f'BeatTime="{total_beats}" />\n'
        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'
        '</WarpMarkers>'
    )

    xml = re.sub(
        r'<WarpMarkers>.*?</WarpMarkers>',
        new_markers,
        xml,
        flags=re.DOTALL,
    )
    return xml


def disable_clip_loop_raw(xml):
    """Disable clip looping (inside Loop blocks), preserve Transport LoopOn."""
    # Only disable LoopOn inside <Loop>...</Loop> blocks (clip loops),
    # not the Transport LoopOn which controls the arrangement loop brace.
    def disable_loop_in_block(m):
        return re.sub(r'(<LoopOn\s+Value=")[^"]*(")', r'\g<1>false\2', m.group(0))
    return re.sub(r'<Loop>.*?</Loop>', disable_loop_in_block, xml, flags=re.DOTALL)


def set_clip_length_raw(xml, beat_length):
    """Set clip CurrentEnd, LoopEnd, and OutMarker to the given beat length."""
    beat_str = f"{beat_length:.6g}"
    xml = re.sub(r'(<CurrentEnd\s+Value=")[^"]*(")', rf'\g<1>{beat_str}\2', xml)
    xml = re.sub(r'(<OutMarker\s+Value=")[^"]*(")', rf'\g<1>{beat_str}\2', xml)
    # LoopEnd inside Loop blocks
    def set_loop_end_in_block(m):
        return re.sub(r'(<LoopEnd\s+Value=")[^"]*(")', rf'\g<1>{beat_str}\2', m.group(0))
    xml = re.sub(r'<Loop>.*?</Loop>', set_loop_end_in_block, xml, flags=re.DOTALL)
    return xml


def set_arrangement_loop_raw(xml, start_beats, length_beats):
    """Set the arrangement loop range in the Transport section."""
    start_str = f"{start_beats:.6g}"
    length_str = f"{length_beats:.6g}"
    def replace_transport(m):
        block = m.group(0)
        block = re.sub(r'(<LoopStart\s+Value=")[^"]*(")', rf'\g<1>{start_str}\2', block)
        block = re.sub(r'(<LoopLength\s+Value=")[^"]*(")', rf'\g<1>{length_str}\2', block)
        return block
    return re.sub(r'<Transport>.*?</Transport>', replace_transport, xml, flags=re.DOTALL)


def insert_audio_clip_raw(xml, wav_path):
    """Replace audio file references in SampleRef/FileRef blocks only.

    Only modifies FileRef elements that are direct children of SampleRef,
    leaving FilePresetRef (device/effect presets) untouched.
    """
    abs_wav = os.path.abspath(wav_path)

    def replace_sample_ref_block(m):
        """Replace Path and RelativePath within a single SampleRef block."""
        block = m.group(0)
        block = re.sub(
            r'(<Path\s+Value=")[^"]*(")',
            lambda _: f'<Path Value="{abs_wav}"',
            block,
        )
        block = re.sub(
            r'(<RelativePath\s+Value=")[^"]*(")',
            lambda _: f'<RelativePath Value="{abs_wav}"',
            block,
        )
        block = re.sub(
            r'(<RelativePathType\s+Value=")[^"]*(")',
            r'\g<1>0\2',
            block,
        )
        return block

    # Only match <SampleRef>...</SampleRef> blocks
    xml = re.sub(
        r'<SampleRef>.*?</SampleRef>',
        replace_sample_ref_block,
        xml,
        flags=re.DOTALL,
    )

    return xml


def get_wav_duration(wav_path):
    """Get the duration of a WAV file in seconds."""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.duration
    except ImportError:
        import wave
        with wave.open(wav_path, 'rb') as w:
            return w.getnframes() / w.getframerate()


def create_project(template_path, wav_path, target_bpm, output_als_path,
                   source_bpm=None):
    """
    Create an Ableton .als project from a template using raw XML manipulation.

    Args:
        template_path: Path to the base .als template
        wav_path: Path to the source WAV file
        target_bpm: Target tempo (BPM) for the project
        output_als_path: Path where the generated .als will be written
        source_bpm: Original BPM of the source file (for clip length calculation)
    """
    if not os.path.exists(template_path):
        print(f"Error: Template not found: {template_path}", file=sys.stderr)
        return False

    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found: {wav_path}", file=sys.stderr)
        return False

    xml = read_als_raw(template_path)

    xml = set_tempo_raw(xml, target_bpm)
    xml = set_warp_mode_raw(xml, WARP_MODE_COMPLEX_PRO)
    xml = set_warping_enabled_raw(xml, enabled=True)
    xml = disable_clip_loop_raw(xml)
    xml = insert_audio_clip_raw(xml, wav_path)

    # Set clip length, warp markers, and arrangement loop for full audio
    if source_bpm:
        wav_duration = get_wav_duration(wav_path)
        clip_beats = wav_duration * source_bpm / 60.0
        xml = set_warp_markers_raw(xml, source_bpm, wav_duration)
        xml = set_clip_length_raw(xml, clip_beats)
        xml = set_arrangement_loop_raw(xml, 0, clip_beats)

    os.makedirs(os.path.dirname(output_als_path) or ".", exist_ok=True)
    write_als_raw(xml, output_als_path)

    print(f"Created project: {output_als_path} "
          f"(tempo={target_bpm} BPM, warp=Complex Pro, clip={os.path.basename(wav_path)})")
    return True


# --- ElementTree-based read-only operations ---

def decompress_als(als_path):
    """Read an .als file (gzip-compressed XML) and return an ElementTree.

    WARNING: ET may drop sibling elements with the same tag. Only use for
    read-only inspection, never for round-trip modification.
    """
    with gzip.open(als_path, "rb") as f:
        raw_xml = f.read()
    return ET.ElementTree(ET.fromstring(raw_xml))


def decompress_als_to_xml(als_path, xml_path):
    """Decompress an .als file to a readable .xml file."""
    with gzip.open(als_path, "rb") as f:
        raw_xml = f.read()
    with open(xml_path, "wb") as f:
        f.write(raw_xml)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ableton .als projects from a template"
    )
    subparsers = parser.add_subparsers(dest="command")

    # create subcommand
    create_parser = subparsers.add_parser("create", help="Create a new .als project")
    create_parser.add_argument("template", help="Path to base .als template")
    create_parser.add_argument("wav", help="Path to source WAV file")
    create_parser.add_argument("bpm", type=float, help="Target tempo (BPM)")
    create_parser.add_argument("output", help="Output .als path")
    create_parser.add_argument("--source-bpm", type=float, default=None,
                               help="Source BPM (for clip length calculation)")

    # decompress subcommand
    decomp_parser = subparsers.add_parser("decompress", help="Decompress .als to .xml")
    decomp_parser.add_argument("als", help="Input .als file")
    decomp_parser.add_argument("xml", help="Output .xml file")

    # compress subcommand
    comp_parser = subparsers.add_parser("compress", help="Compress .xml to .als")
    comp_parser.add_argument("xml", help="Input .xml file")
    comp_parser.add_argument("als", help="Output .als file")

    # inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Show project settings")
    inspect_parser.add_argument("als", help="Input .als file")

    args = parser.parse_args()

    if args.command == "create":
        success = create_project(args.template, args.wav, args.bpm, args.output,
                                source_bpm=args.source_bpm)
        sys.exit(0 if success else 1)

    elif args.command == "decompress":
        decompress_als_to_xml(args.als, args.xml)
        print(f"Decompressed: {args.als} -> {args.xml}")

    elif args.command == "compress":
        with open(args.xml, "r", encoding="utf-8") as f:
            xml = f.read()
        write_als_raw(xml, args.als)
        print(f"Compressed: {args.xml} -> {args.als}")

    elif args.command == "inspect":
        tree = decompress_als(args.als)
        root = tree.getroot()

        # Show tempo
        for tempo in root.iter("Tempo"):
            manual = tempo.find("Manual")
            if manual is not None:
                print(f"Tempo: {manual.get('Value')} BPM")

        # Show warp modes
        for i, wm in enumerate(root.iter("WarpMode")):
            mode_names = {0: "Beats", 1: "Tones", 2: "Texture",
                          3: "Re-Pitch", 4: "Complex", 6: "Complex Pro"}
            val = int(wm.get("Value", -1))
            print(f"WarpMode[{i}]: {val} ({mode_names.get(val, 'Unknown')})")

        # Show audio file references
        for i, sr in enumerate(root.iter("SampleRef")):
            fr = sr.find("FileRef")
            if fr is not None:
                # Try Name first, fall back to Path or RelativePath
                name = fr.find("Name")
                path = fr.find("Path")
                rel_path = fr.find("RelativePath")
                if name is not None and name.get("Value"):
                    print(f"AudioClip[{i}]: {name.get('Value')}")
                elif path is not None and path.get("Value"):
                    print(f"AudioClip[{i}]: {path.get('Value')}")
                elif rel_path is not None and rel_path.get("Value"):
                    print(f"AudioClip[{i}]: {rel_path.get('Value')}")

        # Show loop state
        for i, lo in enumerate(root.iter("LoopOn")):
            print(f"LoopOn[{i}]: {lo.get('Value')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
