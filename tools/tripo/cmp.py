"""Stack the r_<view>_<tripo|mpfb>.png renders into comparison sheets (top the input head, bottom MPFB)."""
import sys
from PIL import Image, ImageDraw
d = sys.argv[1]
for k, r in enumerate([['front', 'q35', 'side', 'back'], ['closeF', 'closeS']]):
    W = Image.new('RGB', (512 * len(r), 1024 + 30), 'white'); D = ImageDraw.Draw(W)
    for i, v in enumerate(r):
        for j, w in enumerate(['tripo', 'mpfb']):
            W.paste(Image.open(f'{d}/r_{v}_{w}.png').convert('RGB').resize((512, 512)), (i * 512, 30 + j * 512))
    D.text((8, 8), 'top: input   bottom: MPFB fitted + input texture', fill='black')
    W.save(f'{d}/cmp_{k}.jpg', quality=92)
