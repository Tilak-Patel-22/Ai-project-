from flask import Flask, render_template, request, jsonify
import os
import csv
import librosa
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load songs from CSV
song_database = []
with open('songs.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        song_database.append({
            "title": row["title"],
            "artist": row["artist"],
            "range": row["range"]
        })

# Real pitch detection using librosa
def analyze_pitch(filepath):
    y, sr = librosa.load(filepath, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)

    if not pitch_values:
        return 0  # fallback
    return np.mean(pitch_values)

# Map pitch to vocal range
def estimate_vocal_range(avg_pitch):
    if avg_pitch == 0:
        return "Unknown"
    elif avg_pitch > 1000:
        return "Soprano"
    elif avg_pitch > 440:
        return "Alto"
    elif avg_pitch > 220:
        return "Tenor"
    else:
        return "Bass"

# Suggest songs based on vocal range
def suggest_songs(vocal_range):
    return [song for song in song_database if song["range"].lower() == vocal_range.lower()]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio_data']
    filename = secure_filename(audio_file.filename)
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", filename)
    audio_file.save(temp_path)

    try:
        avg_pitch = analyze_pitch(temp_path)
        vocal_range = estimate_vocal_range(avg_pitch)
        songs = suggest_songs(vocal_range)

        return jsonify({
            "vocal_range": vocal_range,
            "avg_pitch": round(avg_pitch, 2),
            "songs": songs
        })

    except Exception as e:
        import traceback
        print("Error:", traceback.format_exc())  # Log detailed error in terminal
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    app.run(debug=True)
