"""Read and replace images and mesh attributes in a .vrm/.glb without Blender.

read:  load(path) -> (json, bin); image(json, bin, name) -> bytes; attr(json, bin, mesh, key) -> np.ndarray
write: save(path, json, bin, {image_name: png_bytes}) re-lays the binary chunk with the new images.
"""
import json
import struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
SIZE = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load(path):
    f = open(path, 'rb').read()
    n = struct.unpack('<I', f[12:16])[0]
    j = json.loads(f[20:20 + n])
    bl = struct.unpack('<I', f[20 + n:24 + n])[0]
    return j, f[28 + n:28 + n + bl]


def view(j, b, i):
    bv = j['bufferViews'][i]
    o = bv.get('byteOffset', 0)
    return b[o:o + bv['byteLength']]


def image(j, b, name):
    im = next(im for im in j['images'] if im.get('name') == name)
    return view(j, b, im['bufferView'])


def accessor(j, b, i):
    a = j['accessors'][i]
    bv = j['bufferViews'][a['bufferView']]
    dt, k = COMP[a['componentType']], SIZE[a['type']]
    o = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = bv.get('byteStride', 0) or np.dtype(dt).itemsize * k
    raw = np.frombuffer(b, np.uint8, count=stride * (a['count'] - 1) + np.dtype(dt).itemsize * k, offset=o)
    out = np.lib.stride_tricks.as_strided(raw, (a['count'], np.dtype(dt).itemsize * k), (stride, 1)).copy()
    return out.view(dt).reshape(a['count'], k)


def attr(j, b, mesh, key):
    m = next(m for m in j['meshes'] if m['name'] == mesh)
    return accessor(j, b, m['primitives'][0]['attributes'][key])


def save(path, j, b, images):
    j = json.loads(json.dumps(j))
    repl = {}
    for im in j['images']:
        if im.get('name') in images:
            repl[im['bufferView']] = images[im['name']]
            im['mimeType'] = 'image/png'
    out = bytearray()
    for i, bv in enumerate(j['bufferViews']):
        data = repl.get(i, view(j, b, i))
        while len(out) % 4:
            out.append(0)
        bv['byteOffset'] = len(out)
        bv['byteLength'] = len(data)
        out += data
    while len(out) % 4:
        out.append(0)
    j['buffers'][0]['byteLength'] = len(out)
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(out)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I4s', len(js), b'JSON') + js)
        f.write(struct.pack('<I4s', len(out), b'BIN\x00') + bytes(out))
