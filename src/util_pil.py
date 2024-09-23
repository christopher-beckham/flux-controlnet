import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter

def superimpose(img1, img2, which_alpha="merge"):
    assert img1.mode == "RGBA"
    assert img2.mode == "RGBA"
    assert which_alpha in ["merge", "img1", "img2"]
    buf = img1.copy()
    buf.alpha_composite(img2)
    if which_alpha == "merge":
        pass
    elif which_alpha == "img1":
        buf.putalpha(img1.split()[-1])
    else:
        buf.putalpha(img2.split()[-1])
    return buf

def make_alpha_all_ones(img):
    """Make a version of img where the alpha channel is all-white."""
    assert img.mode == 'RGBA'
    background = Image.new('RGBA', img.size, (255,255,255))
    alpha_composite = Image.alpha_composite(background, img)
    return alpha_composite

def composite(masks):
    """Flatten a list of RGBA masks into a single RGBA image."""
    assert len(masks) > 0, "masks array cannot be zero length"
    canvas = None
    for idx in range(len(masks)):
        if canvas is None:
            canvas = masks[idx].copy()
        else:
            canvas.alpha_composite(masks[idx])
    return canvas


def put_alpha(img, alpha):
    assert alpha.mode == "L"
    img2 = img.copy()
    img2.putalpha(alpha)
    return img2


def replace_alpha_with_white(img_rgba):
    assert img_rgba.mode == 'RGBA'
    background = Image.new("RGBA", img_rgba.size, (255, 255, 255))
    return Image.alpha_composite(background, img_rgba)

def clean_image(pil_img):
    """This method is needed because some RGBA images are 'dirty',
    in the sense that their rgb values may be non-(0,0,0) even
    if their corresponding value in the alpha channel is 0. We
    don't want this discrepency to manifest elsewhere as a silent
    bug so it's best we nip it here.

    This returns RGB images.
    """
    if pil_img.mode == 'RGBA':
        mask = np.array(pil_img.split()[-1])
        img = np.array(pil_img)
        img[ mask == 0 ] = [0,0,0,0]
        return Image.fromarray(img).convert("RGB")
    elif pil_img.mode in ['RGB', 'L']:
        return pil_img.convert("RGB")
    else:
        raise NotImplementedError

def samples_to_rows(
    samples,
    resize_to: int = 512,
    include_cond_image: bool = True,
    num_generated_images: int = None,
    title: str = None,
    title_fontsize: int = None
):
    """
    Produce one large PIL image where each row is a validation image
      and each column is a generation.
    """
    all_rows = []
    for b in range(len(samples)):
        gen_images = samples[b]['images']
        if num_generated_images is not None:
            num_generated_images = min(num_generated_images, len(gen_images))
        else:
            num_generated_images = len(gen_images)
        #this_outdir = os.path.join(out_dir, str(b))
        #print(b, this_outdir)
        if include_cond_image:
            canvas_img = samples[b]['cond_image'].resize((resize_to, resize_to))
            start_at_idx = 0
        else:
            canvas_img = gen_images[0].resize((resize_to, resize_to))
            start_at_idx = 1
        for j in range(start_at_idx, num_generated_images):
            canvas_img = get_concat_h(
                canvas_img, 
                gen_images[j].resize((resize_to, resize_to))
            )  
        all_rows.append(canvas_img)

    return all_rows

    """
    full_img = foldl(get_concat_v, all_rows)
    if title is not None:
        if title_fontsize is None:
            title_fontsize = full_img.width / len(title)
        full_img = draw_text(full_img, title, fontsize=title_fontsize)

    return full_img
    """

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def draw_text(img, text, fontsize=30, pad_color=(255,255,255)):
    if "FONT_DIR" not in os.environ:
        raise ValueError("FONT_DIR needs to be defined")
    font_dir = os.environ["FONT_DIR"]
    img =  add_margin(img, fontsize+5, 0, 0, 0, color=pad_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        font="{}/RobotoMono-Regular.ttf".format(font_dir), 
        size=fontsize
    )
    # font = ImageFont.truetype(<font-file>, <font-size>)
    #font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0), text, (0,0,0), font=font)
    return img

def get_concat_h(im1, im2, resize=None):
    if resize is not None:
        im1 = im1.resize(resize)
        im2 = im2.resize(resize)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2, resize=None):
    if resize is not None:
        im1 = im1.resize(resize)
        im2 = im2.resize(resize)    
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def threshold(img, thresh):
    return Image.fromarray( 
        ((np.array(img) > thresh).astype(np.float32)*255.).astype(np.uint8)
    )