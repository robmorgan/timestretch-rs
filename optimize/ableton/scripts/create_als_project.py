#!/usr/bin/env python3
"""
Generate Ableton Live .als project files from a template.

Manipulates the ALS XML to set tempo, warp mode (Complex Pro), and insert
an audio clip reference. The generated .als can be opened in Ableton for
rendering.

Usage:
    python create_als_project.py template.als input.wav 115.0 output.als
    python create_als_project.py --decompress template.als template.xml
"""
import argparse
import gzip
import os
import sys
import xml.etree.ElementTree as ET


# Ableton warp modes:
#   0 = Beats, 1 = Tones, 2 = Texture, 3 = Re-Pitch, 4 = Complex, 6 = Complex Pro
WARP_MODE_COMPLEX_PRO = 6


def decompress_als(als_path):
    """Read an .als file (gzip-compressed XML) and return an ElementTree."""
    with gzip.open(als_path, "rb") as f:
        raw_xml = f.read()
    return ET.ElementTree(ET.fromstring(raw_xml))


def compress_als(tree, als_path):
    """Write an ElementTree back to a gzip-compressed .als file."""
    xml_bytes = ET.tostring(tree.getroot(), encoding="unicode", xml_declaration=False)
    # Ableton ALS files use XML declaration with version 1.0
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    full_xml = header + xml_bytes
    with gzip.open(als_path, "wb") as f:
        f.write(full_xml.encode("utf-8"))


def decompress_als_to_xml(als_path, xml_path):
    """Decompress an .als file to a readable .xml file."""
    with gzip.open(als_path, "rb") as f:
        raw_xml = f.read()
    with open(xml_path, "wb") as f:
        f.write(raw_xml)


def set_tempo(tree, bpm):
    """Set the master tempo in the Ableton project."""
    # Path: Ableton/LiveSet/MasterTrack/DeviceChain/Mixer/Tempo/Manual
    # Also: Ableton/LiveSet/MasterTrack/AutomationEnvelopes may have tempo
    root = tree.getroot()

    # Primary tempo location
    for tempo_manual in root.iter("Tempo"):
        manual = tempo_manual.find("Manual")
        if manual is not None:
            manual.set("Value", str(float(bpm)))
            return True

    # Fallback: search for any Tempo Manual element
    for elem in root.iter():
        if elem.tag == "Manual" and elem.get("Value"):
            parent = None
            for p in root.iter():
                if elem in list(p):
                    parent = p
                    break
            if parent is not None and parent.tag == "Tempo":
                elem.set("Value", str(float(bpm)))
                return True

    print("Warning: Could not find Tempo element in ALS template.", file=sys.stderr)
    return False


def set_warp_mode(tree, mode=WARP_MODE_COMPLEX_PRO):
    """Set warp mode on all audio clips in the project."""
    root = tree.getroot()
    found = False
    for warp_mode in root.iter("WarpMode"):
        warp_mode.set("Value", str(mode))
        found = True
    if not found:
        print("Warning: No WarpMode elements found in ALS template.", file=sys.stderr)
    return found


def set_warping_enabled(tree, enabled=True):
    """Enable or disable warping on all audio clips."""
    root = tree.getroot()
    value = "true" if enabled else "false"
    for warp_elem in root.iter("IsWarped"):
        warp_elem.set("Value", value)
    for warp_elem in root.iter("Warped"):
        warp_elem.set("Value", value)


def insert_audio_clip(tree, wav_path):
    """
    Set the audio file reference in the first audio clip found.

    This modifies the SampleRef element(s) to point at the given WAV file.
    The wav_path should be an absolute path.
    """
    root = tree.getroot()
    abs_wav = os.path.abspath(wav_path)
    filename = os.path.basename(abs_wav)
    directory = os.path.dirname(abs_wav)

    found = False
    for sample_ref in root.iter("SampleRef"):
        file_ref = sample_ref.find("FileRef")
        if file_ref is not None:
            # Set the file path components
            name_elem = file_ref.find("Name")
            if name_elem is not None:
                name_elem.set("Value", filename)

            # Ableton uses different path representations
            # HasRelativePath, RelativePath, Path, Type
            path_elem = file_ref.find("Path")
            if path_elem is not None:
                path_elem.set("Value", abs_wav)

            # Set absolute path hint
            search_hint = file_ref.find("SearchHint")
            if search_hint is not None:
                path_hint = search_hint.find("PathHint")
                if path_hint is not None:
                    # Clear existing path hint entries and add new ones
                    for child in list(path_hint):
                        path_hint.remove(child)

            # Set RelativePathType to indicate absolute
            rel_type = file_ref.find("RelativePathType")
            if rel_type is not None:
                rel_type.set("Value", "0")  # 0 = absolute

            # HasRelativePath
            has_rel = file_ref.find("HasRelativePath")
            if has_rel is not None:
                has_rel.set("Value", "false")

            # LivePackName / LivePackId (clear them)
            for tag in ("LivePackName", "LivePackId"):
                elem = file_ref.find(tag)
                if elem is not None:
                    elem.set("Value", "")

            found = True

    if not found:
        print("Warning: No SampleRef/FileRef found in ALS template. "
              "Ensure the template has at least one audio clip.", file=sys.stderr)

    return found


def disable_normalization(tree):
    """Disable any audio normalization settings."""
    root = tree.getroot()
    for elem in root.iter("SampleVolume"):
        manual = elem.find("Manual")
        if manual is not None:
            manual.set("Value", "1")


def create_project(template_path, wav_path, target_bpm, output_als_path):
    """
    Create an Ableton .als project from a template.

    Args:
        template_path: Path to the base .als template
        wav_path: Path to the source WAV file
        target_bpm: Target tempo (BPM) for the project
        output_als_path: Path where the generated .als will be written
    """
    if not os.path.exists(template_path):
        print(f"Error: Template not found: {template_path}", file=sys.stderr)
        return False

    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found: {wav_path}", file=sys.stderr)
        return False

    tree = decompress_als(template_path)

    set_tempo(tree, target_bpm)
    set_warp_mode(tree, WARP_MODE_COMPLEX_PRO)
    set_warping_enabled(tree, enabled=True)
    insert_audio_clip(tree, wav_path)
    disable_normalization(tree)

    os.makedirs(os.path.dirname(output_als_path) or ".", exist_ok=True)
    compress_als(tree, output_als_path)

    print(f"Created project: {output_als_path} "
          f"(tempo={target_bpm} BPM, warp=Complex Pro, clip={os.path.basename(wav_path)})")
    return True


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
        success = create_project(args.template, args.wav, args.bpm, args.output)
        sys.exit(0 if success else 1)

    elif args.command == "decompress":
        decompress_als_to_xml(args.als, args.xml)
        print(f"Decompressed: {args.als} -> {args.xml}")

    elif args.command == "compress":
        tree = ET.parse(args.xml)
        compress_als(tree, args.als)
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
                name = fr.find("Name")
                if name is not None:
                    print(f"AudioClip[{i}]: {name.get('Value', '?')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
