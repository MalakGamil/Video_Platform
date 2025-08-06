

# AI-Powered Video Generation Platform


Transform your text documents and PDFs into engaging, professionally-styled videos with the power of AI. This platform uses Google's Gemini API to analyze your content, generate a slide-by-slide narrative, and automatically produce a video complete with voiceover, images, and elegant design.

## ✨ Key Features

-   **Multi-Format Support**: Upload `.txt` or `.pdf` files to generate videos.
-   **AI-Powered Content Generation**: Leverages the Gemini 1.5 Flash model to create compelling titles, detailed explanations, and concise speech text for each slide.
-   **Automated Visuals**: Intelligently searches for and integrates high-quality, relevant images from the Pixabay API.
-   **Dynamic Slide Design**: Each slide is beautifully designed with modern fonts, elegant backgrounds, and smooth transitions.
-   **Text-to-Speech (TTS)**: Automatically generates a natural-sounding voiceover for each slide, with support for both English and Arabic.
-   **Web-Based Interface**: A simple, intuitive, and stylish web interface for easy file uploads and video generation.
-   **RESTful API**: A clean backend API built with Flask.

## ⚙️ How It Works

The project follows a simple yet powerful workflow:

1.  **File Upload**: The user uploads a `.txt` or `.pdf` file through the web interface.
2.  **Text Extraction**: The Flask backend receives the file and extracts its text content.
3.  **AI Slide Generation**: The extracted text is sent to the Google Gemini API, which returns a structured JSON array of slide data (title, explanation, image query, etc.).
4.  **Image & Audio Creation**: For each slide, the system:
    -   Downloads a relevant image from Pixabay based on the AI-generated query.
    -   Generates an MP3 audio file from the slide's text using gTTS.
5.  **Video Assembly**: The `moviepy` library assembles the generated images and audio files into video clips.
6.  **Final Composition**: All individual clips are concatenated into a final MP4 video.
7.  **Display**: The final video is presented to the user in the web interface for viewing and downloading.

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

Make sure you have the following installed on your system:
-   [Python 3.8+](https://www.python.org/downloads/)
-   `pip` (Python package installer)
-   `virtualenv` (Recommended for creating isolated Python environments)
    ```bash
    pip install virtualenv
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    -   **On Windows:**
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install the required dependencies:**
    Use the provided `requirements.txt` file to install all necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    You need API keys from Google (for Gemini) and Pixabay.
    -   Find the `video_generator.py` file in the `src/routes/` directory.
    -   Replace the placeholder keys with your actual API keys:
        -   In the `generate_slide_data` function, find `genai.configure(api_key="YOUR_API_KEY")`.
        -   In the `download_pixabay_image` function, find `API_KEY = "YOUR_API_KEY"`.

### Running the Application

Once the installation is complete, you can run the application with a single command from the project's root directory:

```bash
python src/main.py
```

The application will start, and you can access it by opening your web browser and navigating to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## 📖 How to Use

1.  **Open the Web Interface**: Go to `http://127.0.0.1:5000` in your browser.
2.  **Upload a File**:
    -   Click the upload area to select a `.txt` or `.pdf` file from your computer.
    -   Or, drag and drop your file directly into the upload area.
3.  **Generate Video**: Once the file is uploaded, the "Generate Video" button will be enabled. Click it to start the process.
4.  **Wait for Generation**: The process may take a few minutes as it involves API calls, image downloads, and video rendering. A loading spinner will indicate that the application is working.
5.  **View and Download**: When the video is ready, it will appear on the page. You can play it directly or use the controls to download it to your device.

## 📂 Project Structure

```
.
├── .venv/                  # Virtual environment directory
├── src/
│   ├── static/             # Frontend files (HTML, CSS, JS)
│   │   └── index.html
│   ├── templates/          # (Optional) For server-side templates
│   ├── routes/
│   │   └── video_generator.py  # Core logic for video generation
│   ├── models/             # (Optional) For database models
│   └── main.py             # Main Flask application entry point
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```




