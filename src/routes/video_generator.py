import os
import random
import colorsys
import re
import json
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


def hex_to_rgb(hex_color):
    """Converts a hex color string to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Converts an (r, g, b) tuple to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)


def get_contrasting_color(hex_color):
    """Get a contrasting color (black or white) for text."""
    r, g, b = hex_to_rgb(hex_color)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#FFFFFF" if brightness < 128 else "#000000"


def get_vibrant_title_color(hex_color):
    """Get a vibrant, contrasting color for titles."""
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    new_hue = (h + 0.5) % 1.0
    new_saturation = max(0.8, s)
    new_value = max(0.9, v)
    new_r, new_g, new_b = [int(c * 255) for c in colorsys.hsv_to_rgb(new_hue, new_saturation, new_value)]
    return rgb_to_hex((new_r, new_g, new_b))


def create_gradient(base_color, size):
    """Creates a vertical gradient background from the base color."""
    r, g, b = hex_to_rgb(base_color)
    gradient = Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for y in range(size[1]):
        ratio = y / size[1]
        new_r = int(r * (1 - ratio * 0.2))
        new_g = int(g * (1 - ratio * 0.2))
        new_b = int(b * (1 - ratio * 0.2))
        draw.line([(0, y), (size[0], y)], fill=(new_r, new_g, new_b, 255))
    return gradient


def add_background_patterns(draw, size, base_color):
    """Adds subtle, randomly generated geometric patterns to the background."""
    r, g, b = hex_to_rgb(base_color)
    pattern_color = (min(r + 25, 255), min(g + 25, 255), min(b + 25, 255), 40)
    for _ in range(50):
        shape_type = random.choice(["circle", "triangle", "line"])
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        if shape_type == "circle":
            radius = random.randint(30, 150)
            draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=pattern_color, outline=None)
        elif shape_type == "triangle":
            x2, y2 = x1 + random.randint(-120, 120), y1 + random.randint(-120, 120)
            x3, y3 = x1 + random.randint(-120, 120), y1 + random.randint(-120, 120)
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=pattern_color, outline=None)
        elif shape_type == "line":
            x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
            width = random.randint(2, 5)
            draw.line([(x1, y1), (x2, y2)], fill=pattern_color, width=width)


def add_decorative_frame(draw, size):
    """Adds a decorative frame around the slide."""
    frame_color = (255, 255, 255, 80)
    margin = 30
    draw.rectangle([(margin, margin), (size[0] - margin, size[1] - margin)], outline=frame_color, width=3)
    draw.rectangle([(margin + 8, margin + 8), (size[0] - margin - 8, size[1] - margin - 8)], outline=frame_color,
                   width=2)
    corner_size = 45
    for x in [margin, size[0] - margin - corner_size]:
        for y in [margin, size[1] - margin - corner_size]:
            draw.rectangle([(x, y), (x + corner_size, y + corner_size)], outline=frame_color, width=2)


# --- Main Function to Create Slide ---

def create_slide_image(slide, index, image_path=None, work_dir="/tmp"):
    """
    Creates a single slide image with an expanded text area.
    The slide splitting logic has been removed.
    """
    bg_color = slide.get("background", "#1E1E3F")
    width, height = 1920, 1150

    # 1. Create Base Image
    img = create_gradient(bg_color, (width, height)).convert("RGBA")
    pattern_draw = ImageDraw.Draw(img)
    add_background_patterns(pattern_draw, (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("LATINWD.TTF", size=60)
        text_font = ImageFont.truetype("MOD20.TTF", size=38)
        number_font = ImageFont.truetype("LATINWD.TTF", size=52)
    except IOError:
        print("Custom fonts not found. Using default fonts.")
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    # 2. Draw Title
    title_text = slide.get("title", "Default Title")
    title_color = get_vibrant_title_color(bg_color)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = (width - title_width) / 2
    title_y = 70
    draw.text((title_x, title_y), title_text, font=title_font, fill=title_color)
    underline_y = title_y + title_height + 10
    draw.line([(title_x, underline_y), (title_x + title_width, underline_y)], fill=title_color, width=5)

    # 3. Draw Explanation Text
    explanation = slide.get("explanation", "")
    if explanation:
        text_area_x, text_area_y = 90, 290
        max_text_width = 870

        # Wrap text logic remains the same
        words = explanation.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            line_bbox = draw.textbbox((0, 0), test_line, font=text_font)
            if line_bbox[2] - line_bbox[0] <= max_text_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        line_height = text_font.getbbox("A")[3] * 1.8
        text_color = get_contrasting_color(bg_color)

        max_text_area_height = height - text_area_y - 100

        for i, line in enumerate(lines):

            if (i * line_height) > max_text_area_height:
                break
            draw.text((text_area_x, text_area_y + i * line_height), line, font=text_font, fill=text_color)

    if image_path and os.path.exists(image_path):
        try:
            slide_image = Image.open(image_path).convert("RGBA")
            slide_image.thumbnail((800, 700))
            frame_size = 20
            bordered_image = Image.new("RGBA",
                                       (slide_image.width + frame_size * 2, slide_image.height + frame_size * 2),
                                       (255, 255, 255, 255))
            bordered_image.paste(slide_image, (frame_size, frame_size))
            shadow = Image.new("RGBA", (bordered_image.width + 30, bordered_image.height + 30), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rectangle((0, 0, shadow.width, shadow.height), fill=(0, 0, 0, 100))
            shadow = shadow.filter(ImageFilter.GaussianBlur(15))
            img_x, img_y = 1020, 320
            img.paste(shadow, (img_x - 15, img_y + 15), shadow)
            img.paste(bordered_image, (img_x, img_y), bordered_image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    add_decorative_frame(draw, (width, height))
    number_text = str(index + 1)
    number_color = get_contrasting_color(bg_color)
    number_bbox = draw.textbbox((0, 0), number_text, font=number_font)
    number_width = number_bbox[2] - number_bbox[0]
    draw.text((width - 70 - number_width / 2, height - 100), number_text, font=number_font, fill=number_color)

    final_img = img.convert("RGB")
    path = os.path.join(work_dir, f"slide_{index}.png")
    final_img.save(path)
    return path


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
