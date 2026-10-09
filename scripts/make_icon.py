from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent.parent
icon_dir = root / 'src-tauri' / 'icons'
icon_dir.mkdir(parents=True, exist_ok=True)
public_dir = root / 'frontend' / 'public'
public_dir.mkdir(parents=True, exist_ok=True)

size = 1024
background = Image.new('RGBA', (size, size), (3, 7, 35, 255))
draw = ImageDraw.Draw(background)

outer_mask = Image.new('L', (size, size), 0)
outer_draw = ImageDraw.Draw(outer_mask)
outer_draw.rounded_rectangle((210, 165, 690, 850), radius=105, fill=255)
outer_draw.ellipse((400, 165, 900, 850), fill=255)

gradient = Image.new('RGBA', (size, size), (0, 0, 0, 0))
gradient_draw = ImageDraw.Draw(gradient)
stops = [(0.0, (0, 230, 238)), (0.5, (0, 74, 255)), (1.0, (169, 35, 255))]
for x in range(190, 921):
	position = (x - 190) / 730
	for index in range(len(stops) - 1):
		start, start_color = stops[index]
		end, end_color = stops[index + 1]
		if position <= end:
			amount = (position - start) / (end - start)
			color = tuple(round(a + (b - a) * amount) for a, b in zip(start_color, end_color))
			break
	gradient_draw.line((x, 0, x, size), fill=(*color, 255))
background = Image.composite(gradient, background, outer_mask)

draw = ImageDraw.Draw(background)
inner_mask = Image.new('L', (size, size), 0)
inner_draw = ImageDraw.Draw(inner_mask)
inner_draw.rounded_rectangle((338, 260, 605, 755), radius=25, fill=255)
inner_draw.ellipse((390, 260, 772, 755), fill=255)
inner_fill = Image.new('RGBA', (size, size), (3, 7, 35, 255))
background = Image.composite(inner_fill, background, inner_mask)
draw = ImageDraw.Draw(background)

for y in (270, 385, 500, 615):
	draw.rounded_rectangle((239, y, 311, y + 78), radius=20, fill=(3, 7, 35, 255))

wave_bars = [(370, 92), (414, 138), (458, 196), (502, 252), (546, 184), (590, 126), (634, 70)]
wave_colors = [(0, 230, 238), (0, 204, 250), (0, 166, 255), (40, 111, 255), (76, 78, 255), (124, 57, 255), (175, 47, 255)]
center_y = 405
for (x, height), color in zip(wave_bars, wave_colors):
	draw.rounded_rectangle((x, center_y - height // 2, x + 30, center_y + height // 2), radius=15, fill=(*color, 255))

draw.arc((505, 525, 734, 755), start=205, end=275, fill=(0, 219, 238, 255), width=15)
draw.polygon([(528, 687), (559, 699), (537, 720)], fill=(0, 219, 238, 255))
draw.arc((526, 548, 752, 776), start=25, end=95, fill=(39, 119, 255, 255), width=15)
draw.polygon([(735, 621), (704, 607), (726, 587)], fill=(39, 119, 255, 255))

draw.rounded_rectangle((505, 565, 635, 695), radius=22, fill=(0, 211, 231, 255))
draw.rounded_rectangle((605, 640, 735, 770), radius=22, fill=(112, 36, 239, 255))

try:
	chinese_font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 82)
	font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 92)
except OSError:
	chinese_font = ImageFont.load_default()
	font = ImageFont.load_default()
glyph_box = draw.textbbox((0, 0), '文', font=chinese_font)
draw.text((570 - (glyph_box[2] - glyph_box[0]) / 2, 602 - glyph_box[1]), '文', font=chinese_font, fill=(3, 7, 35, 255))
draw.text((646, 642), 'A', font=font, fill=(255, 255, 255, 255))

image = background.resize((1024, 1024), Image.Resampling.LANCZOS)
image.save(public_dir / 'dubvibe-icon.png', optimize=True)
image.save(icon_dir / 'icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print('ICON_OK')
