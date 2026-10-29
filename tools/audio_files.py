#!/usr/bin/env python3
"""Float-WAV helpers and current concealed listening report entry point.

Shared by generated-signal qualification and listening tools.
"""
import argparse, array, hashlib, html, json, math
from pathlib import Path
import struct, subprocess, tempfile


def wav_read(path):
    data=path.read_bytes();offset=12;fmt=None;payload=None
    while offset+8<=len(data):
        kind,n=struct.unpack_from('<4sI',data,offset);offset+=8
        if kind==b'fmt ':fmt=struct.unpack_from('<HHI',data,offset)
        if kind==b'data':payload=data[offset:offset+n]
        offset+=n+(n%2)
    assert fmt and fmt[0] in (3,65534) and payload is not None,(path,fmt)
    samples=array.array('f');samples.frombytes(payload)
    return fmt[1],fmt[2],samples


def wav_write(path,ch,rate,samples):
    samples=array.array('f',samples);payload=samples.tobytes()
    path.write_bytes(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(payload),b'WAVE',b'fmt ',16,3,ch,rate,rate*ch*4,ch*4,32,b'data',len(payload))+payload)
