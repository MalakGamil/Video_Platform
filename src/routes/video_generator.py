import os
# import random
import re
import json
# import textwrap
import requests
import shutil
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from flask import Blueprint, request, jsonify, current_app
# from werkzeug.utils import secure_filename
# import tempfile
import uuid
import fitz  # PyMuPDF


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

genai.configure(api_key="AIzaSyA7nvrEK25C7ytzX81T1o57kYZIemmyk2M")

video_bp = Blueprint('video', __name__)

def extract_keywords(text, lang='english'):
    """Extract keywords from the specified text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words(lang))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    keywords = [word for word, count in word_counts.most_common(5)]
    return keywords

def generate_slide_data(topic: str) -> list[dict]:
    """
    Generates rich, detailed, and engaging slide data using the Gemini API.
    """
    print(f"Generating slide data for topic: '{topic[:50]}...'")

    prompt = f"""
    You are a world-class educator and master storyteller. Your mission is to transform the following complex topic into a visually stunning and intellectually stimulating presentation for a general audience.

    Topic: "{topic}"

    For each slide, generate a JSON object with the following keys. The final output must be a valid JSON array of these objects.

    - "title": A short, powerful, and curiosity-invoking title (max 5 words).
    - "explanation": A detailed, rich, and inspiring explanation (up to 5000 words). Use a conversational, direct tone. Employ storytelling techniques, analogies, and rhetorical questions to make the content memorable and engaging.
    - "background": A modern and elegant hex color code for the slide's background gradient (e.g., #FF6B6B, #4ECDC4, #45B7D1).
    - "image_query": A concise, effective search query for a high-quality, realistic, and professional image that visually represents the slide's core idea.
    - "speech_text": A concise, spoken-word version of the explanation (1-3 sentences). Write it to sound like a passionate and natural TED Talk speaker.

    Guidelines:
    - The number of slides should be appropriate to cover the topic comprehensively.
    - Each slide must feel like a complete, standalone idea, yet flow seamlessly into the next.
    - Prioritize clarity, creativity, and narrative flow over dry facts.
    - The output MUST be a clean, valid JSON array, without any surrounding text or markdown.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        json_text = re.search(r'```json\s*(\[.*])\s*```', response.text, re.DOTALL)
        if json_text:
            return json.loads(json_text.group(1))

        json_text = re.search(r'(\[.*])', response.text, re.DOTALL)
        if json_text:
            return json.loads(json_text.group(1))

        return json.loads(response.text.strip())

    except Exception as e:
        print(f"Error generating slide data with Gemini: {e}")

        return [
            {
                "title": "An Error Occurred",
                "explanation": f"Failed to generate content for the topic: '{topic}'. The AI model might be unavailable or the topic could not be processed. Please try again later.",
                "background": "#2c3e50",
                "image_query": "system error warning sign",
                "speech_text": "Unfortunately, an error occurred while generating the presentation content. Please check the system and try again."
            }
        ]


def create_slide_image(slide, index, image_path=None, work_dir="/tmp"):
    """Create slide image with improved typography and design"""
    bg_color = slide.get("background", "#1E1E3F")

    img = Image.new("RGB", (1280, 720), color=bg_color)

    gradient = create_gradient(bg_color, (1280, 720))
    img.paste(gradient, (0, 0), gradient)

    img = add_design_elements(img, bg_color)

    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 70)
        number_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
    except:

        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    header_height = 140
    header_bg = Image.new("RGBA", (1280, header_height), color=(0, 0, 0, 200))
    img.paste(header_bg, (0, 0), header_bg)

    slide_number_size = 160
    slide_number_bg = Image.new("RGBA", (slide_number_size, slide_number_size), color=(255, 255, 255, 240))
    mask = Image.new('L', (slide_number_size, slide_number_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, slide_number_size, slide_number_size), fill=255)
    slide_number_bg.putalpha(mask)

    slide_number_draw = ImageDraw.Draw(slide_number_bg)
    text = str(index + 1)
    text_bbox = slide_number_draw.textbbox((0, 0), text, font=number_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (slide_number_size - text_width) // 2
    text_y = (slide_number_size - text_height) // 2
    slide_number_draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=number_font)

    img.paste(slide_number_bg, (1160, 600), slide_number_bg)

    title_text = slide.get("title", "")

    title_color = get_vibrant_title_color(bg_color)

    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (1280 - title_width) // 2
    title_y = 25

    shadow_color = (0, 0, 0, 180)
    draw.text((title_x + 3, title_y + 3), title_text, font=title_font, fill=shadow_color)

    draw.text((title_x, title_y), title_text, font=title_font, fill=title_color)

    explanation = slide.get("explanation", "")

    wrapped_lines = []
    words = explanation.split()
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        test_bbox = draw.textbbox((0, 0), test_line, font=text_font)
        test_width = test_bbox[2] - test_bbox[0]

        if test_width <= 580:  # Max width for text area
            current_line = test_line
        else:
            if current_line:
                wrapped_lines.append(current_line)
            current_line = word

    if current_line:
        wrapped_lines.append(current_line)

    text_area_width = 620
    text_area_height = 320
    text_bg = Image.new("RGBA", (text_area_width, text_area_height), color=(0, 0, 0, 160))
    text_bg = text_bg.filter(ImageFilter.GaussianBlur(radius=4))

    img.paste(text_bg, (40, 170), text_bg)

    line_height = 65
    start_y = 190

    for i, line in enumerate(wrapped_lines[:4]):
        draw.text((60, start_y + i * line_height), line, font=text_font, fill="white")

    if image_path and os.path.exists(image_path):
        try:
            slide_image = Image.open(image_path)
            max_width = 520
            max_height = 420
            slide_image.thumbnail((max_width, max_height))
            frame = create_enhanced_image_frame(slide_image, bg_color)
            img.paste(frame, (680, 160), frame if frame.mode == 'RGBA' else None)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    footer_height = 40
    footer_bg = Image.new("RGBA", (1280, footer_height), color=(0, 0, 0, 180))
    img.paste(footer_bg, (0, 720 - footer_height), footer_bg)

    path = os.path.join(work_dir, f"slide_{index}.png")
    img.save(path)
    return path

def get_vibrant_title_color(hex_color):
    """Get a vibrant color for titles that contrasts well"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    brightness = (r * 299 + g * 587 + b * 114) / 1000

    if brightness < 128:
        vibrant_colors = ["#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFEAA7", "#DDA0DD"]
    else:
        vibrant_colors = ["#8B0000", "#2E8B57", "#4B0082", "#B22222", "#FF4500", "#9932CC"]


    import hashlib
    color_index = int(hashlib.md5(hex_color.encode()).hexdigest(), 16) % len(vibrant_colors)
    return vibrant_colors[color_index]

def create_enhanced_image_frame(image, base_color):
    """Create enhanced frame for image with better styling"""
    frame_width = 12
    frame = Image.new("RGBA", (image.width + 2 * frame_width, image.height + 2 * frame_width),
                      color=(255, 255, 255, 250))


    shadow_offset = 15
    shadow = Image.new("RGBA", (image.width + 2 * frame_width + shadow_offset * 2,
                               image.height + 2 * frame_width + shadow_offset * 2),
                       color=(0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle([(shadow_offset, shadow_offset),
                          (shadow.width - shadow_offset, shadow.height - shadow_offset)],
                         fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12))

    final_frame = Image.new("RGBA", shadow.size, color=(0, 0, 0, 0))
    final_frame.paste(shadow, (0, 0), shadow)
    final_frame.paste(frame, (shadow_offset, shadow_offset), frame)
    final_frame.paste(image, (shadow_offset + frame_width, shadow_offset + frame_width))

    return final_frame

def get_contrasting_color(hex_color, bright=False):
    """Get contrasting color for text"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    brightness = (r * 299 + g * 587 + b * 114) / 1000

    if bright:
        return "#FFFF00" if brightness < 128 else "#FFFFFF"
    else:
        return "#FFFFFF" if brightness < 128 else "#000000"

def create_gradient(base_color, size):
    """Create gradient background"""
    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)

    gradient = Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    for y in range(size[1]):
        alpha = 255 - int(y / size[1] * 200)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b, alpha))

    return gradient

def add_design_elements(img, base_color):
    """Add design elements to image"""
    elements = Image.new("RGBA", img.size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(elements)

    # Add lines
    draw.line([(0, 120), (1280, 120)], fill=(255, 255, 255, 100), width=2)
    draw.line([(0, 690), (1280, 690)], fill=(255, 255, 255, 100), width=2)

    # Add vertical lines
    for i in range(5):
        x = 20 + i * 10
        draw.line([(x, 120), (x, 690)], fill=(255, 255, 255, 30), width=1)

    for i in range(5):
        x = 1260 - i * 10
        draw.line([(x, 120), (x, 690)], fill=(255, 255, 255, 30), width=1)

    return Image.alpha_composite(img.convert("RGBA"), elements).convert("RGB")

def create_image_frame(image, base_color):
    """Create frame for image"""
    frame_width = 10
    frame = Image.new("RGBA", (image.width + 2 * frame_width, image.height + 2 * frame_width),
                      color=(255, 255, 255, 220))

    shadow = Image.new("RGBA", (image.width + 2 * frame_width + 20, image.height + 2 * frame_width + 20),
                       color=(0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle([(10, 10), (shadow.width - 10, shadow.height - 10)], fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))

    final_frame = Image.new("RGBA", shadow.size, color=(0, 0, 0, 0))
    final_frame.paste(shadow, (0, 0), shadow)
    final_frame.paste(frame, (10, 10), frame)
    final_frame.paste(image, (10 + frame_width, 10 + frame_width))

    return final_frame

def create_audio(text, index, work_dir="/tmp", lang="en", voice_type="female"):
    """Create enhanced audio file with more natural speech patterns"""
    audio_path = os.path.join(work_dir, f"audio_{index}.mp3")

    try:
        enhanced_text = enhance_text_for_natural_speech(text)

        tts = gTTS(text=enhanced_text, lang=lang, slow=False, tld='com')
        tts.save(audio_path)

        print(f"Enhanced audio created: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error creating audio: {e}")
        return None

def enhance_text_for_natural_speech(text):
    """Enhanced text processing for more natural, conversational speech"""

    text = text.replace("!", "! ... ")
    text = text.replace("?", "? ... ")
    text = text.replace(". ", ". ... ")
    text = text.replace(", ", ", ... ")
    text = text.replace(": ", ": ... ")

    text = text.replace("AI", "A I")
    text = text.replace("ML", "M L")
    text = text.replace("imagine", "... imagine")
    text = text.replace("think about", "... think about")
    text = text.replace("picture this", "... picture this")
    text = text.replace("here's the thing", "... here's the thing")
    text = text.replace("but wait", "... but wait")
    text = text.replace("now", "... now")

    text = text.replace("Well,", "Well, ... ")
    text = text.replace("So,", "So, ... ")
    text = text.replace("And", "... And")
    text = text.replace("But", "... But")

    return text

def download_pixabay_image(query, index, work_dir="/tmp"):
    """Download image from Pixabay"""
    try:
        API_KEY = "51174532-094a192853086cd93bf32254f"
        print(f"🔍 Searching Pixabay for: {query}")

        search_url = f"https://pixabay.com/api/?key={API_KEY}&q={query}&lang=en&image_type=photo&orientation=horizontal&per_page=10"

        params = {
            'editors_choice': 'true',
            'order': 'popular',
            'safesearch': 'true'
        }

        response = requests.get(search_url, params=params, timeout=15 )

        if response.status_code != 200:
            print(f"⚠️ API Error: {response.status_code}")
            return None

        data = response.json()

        if not data.get('hits'):
            print("⚠️ No images available in Pixabay for this query")
            return None

        selected_photo = max(data['hits'], key=lambda x: x['likes'])
        img_url = selected_photo['largeImageURL']

        print(f"🌄 Found image: {img_url} (Likes: {selected_photo['likes']})")

        img_response = requests.get(img_url, timeout=15)
        if img_response.status_code != 200:
            print("⚠️ Failed to download image")
            return None

        img_path = os.path.join(work_dir, f"slide_{index}.jpg")

        with open(img_path, "wb") as f:
            f.write(img_response.content)

        if os.path.getsize(img_path) < 50 * 1024:
            print("⚠️ Low quality image")
            os.remove(img_path)
            return None

        print(f"✅ Image saved successfully: {img_path}")
        return img_path

    except requests.exceptions.Timeout:
        print("⏰ Timeout while connecting to Pixabay")
        return None
    except Exception as e:
        print(f"🔥 Unexpected error: {str(e)}")
        return None

def build_video(slides_data, work_dir="/tmp"):
    """Build video from slides data"""
    print("Starting video creation...")

    slides_dir = os.path.join(work_dir, "slides")
    audios_dir = os.path.join(work_dir, "audios")
    images_dir = os.path.join(work_dir, "images")
    output_dir = os.path.join(work_dir, "output")

    for dir_path in [slides_dir, audios_dir, images_dir, output_dir]:
        os.makedirs(dir_path, exist_ok=True)

    image_paths = []
    for i, slide in enumerate(slides_data):
        query = slide.get("image_query", "technology")
        image_path = download_pixabay_image(query, i, images_dir)
        image_paths.append(image_path)

    clips = []
    for i, slide in enumerate(slides_data):
        print(f"Processing slide {i + 1}/{len(slides_data)}...")

        image_path = image_paths[i] if i < len(image_paths) else None
        img_path = create_slide_image(slide, i, image_path, slides_dir)

        text_for_audio = slide.get("speech_text", slide.get("explanation", ""))
        lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in text_for_audio) else "en"
        audio_path = create_audio(text_for_audio, i, audios_dir, lang)

        try:
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                duration = max(4, audio_clip.duration + 1)
                clip = ImageClip(img_path).with_duration(duration).with_audio(audio_clip)
            else:
                clip = ImageClip(img_path).with_duration(5)
            clips.append(clip)

        except Exception as e:
            print(f"Error processing slide {i}: {e}")
            clip = ImageClip(img_path).with_duration(5)
            clips.append(clip)

    if not clips:
        print("No clips were generated. Aborting video creation.")
        return None

    print("Combining all slides...")
    final = concatenate_videoclips(clips, method="compose")

    output_path = os.path.join(output_dir, "final_video.mp4")
    final.write_videofile(output_path, fps=24, audio_codec="aac")

    print(f"Video created successfully: {output_path}")
    return output_path

@video_bp.route('/api/generate-video', methods=['POST'])
def generate_video():
    """API endpoint to generate video from uploaded text or PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400

        filename = file.filename
        content = ""

        if filename.endswith('.txt'):
            content = file.read().decode('utf-8').strip()
        elif filename.endswith('.pdf'):
            try:
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf_document:
                    content += page.get_text()
                pdf_document.close()
                content = content.strip()
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error reading PDF file: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'message': 'Only .txt and .pdf files are allowed'}), 400

        if not content:
            return jsonify({'success': False, 'message': 'File is empty or contains no text'}), 400

        work_id = str(uuid.uuid4())
        work_dir = os.path.join(current_app.static_folder, 'temp', work_id)
        os.makedirs(work_dir, exist_ok=True)

        print(f"Processing text from {filename}: {content[:100]}...")

        slides_data = generate_slide_data(content)
        if not slides_data or (len(slides_data) == 1 and "Error Occurred" in slides_data[0].get("title", "")):
             return jsonify({'success': False, 'message': 'Failed to generate slide data from the provided text.'}), 500

        video_path = build_video(slides_data, work_dir)
        if not video_path:
            return jsonify({'success': False, 'message': 'Failed to build the video file.'}), 500

        final_video_name = f"video_{work_id}.mp4"
        final_video_path = os.path.join(current_app.static_folder, final_video_name)
        shutil.copy2(video_path, final_video_path)

        shutil.rmtree(work_dir)

        return jsonify({
            'success': True,
            'message': 'Video generated successfully',
            'video_url': f'/static/{final_video_name}'
        })

    except Exception as e:
        print(f"Error in generate_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'An unexpected error occurred: {str(e)}'}), 500
