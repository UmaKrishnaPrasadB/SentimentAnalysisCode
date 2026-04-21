# import csv
# import os
# from io import StringIO
# from typing import List

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from flask import Flask, jsonify, render_template, request

# from scraper import ReviewScraper
# from utils import HybridSentimentEngine, extract_common_words, summarize_sentiments

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "static"

# engine = HybridSentimentEngine()
# scraper = ReviewScraper()


# def build_chart(counts: dict, filename: str = "sentiment_distribution.png") -> str:
#     labels = ["Positive", "Neutral", "Negative"]
#     values = [counts.get(label, 0) for label in labels]

#     plt.figure(figsize=(8, 4))
#     bars = plt.bar(labels, values, color=["#22c55e", "#facc15", "#ef4444"])
#     plt.title("Sentiment Distribution")
#     plt.xlabel("Sentiment")
#     plt.ylabel("Count")

#     for bar, value in zip(bars, values):
#         plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")

#     output_path = os.path.join("static", filename)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

#     print(f"[DEBUG][app] Chart generated at: {output_path}")
#     return f"/{output_path}"


# def run_inference(texts: List[str]) -> dict:
#     results = [engine.predict_single(text) for text in texts]
#     counts = summarize_sentiments(results)
#     common_words = extract_common_words(texts, top_n=10)
#     chart_path = build_chart(counts)

#     return {
#         "results": results,
#         "counts": counts,
#         "common_words": [{"word": w, "count": c} for w, c in common_words],
#         "chart": chart_path,
#     }


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/api/predict", methods=["POST"])
# def predict_text():
#     try:
#         data = request.get_json(force=True)
#         text = (data.get("text") or "").strip()

#         if not text:
#             return jsonify({"error": "No text provided."}), 400

#         payload = run_inference([text])
#         return jsonify(payload)
#     except Exception as exc:
#         print(f"[ERROR][app] /api/predict failed: {exc}")
#         return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


# @app.route("/api/upload", methods=["POST"])
# def upload_csv():
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part in request."}), 400

#         file = request.files["file"]
#         filename = file.filename.lower()
#         if not (filename.endswith(".csv") or filename.endswith(".txt")):
#             return jsonify({"error": "Only CSV or TXT files are supported."}), 400

#         raw_text = file.stream.read().decode("utf-8", errors="ignore")
#         texts = []

#         if filename.endswith(".csv"):
#             stream = StringIO(raw_text)
#             reader = csv.DictReader(stream)

#             for row in reader:
#                 # expects a column named 'text', fallback to first column
#                 if "text" in row and row["text"]:
#                     texts.append(row["text"])
#                 elif row:
#                     first_value = next(iter(row.values()), "")
#                     if first_value:
#                         texts.append(first_value)
#         else:
#             # TXT input: each non-empty line is treated as a separate review.
#             texts = [line.strip() for line in raw_text.splitlines() if line.strip()]

#         if not texts:
#             return jsonify({"error": "No usable text rows found in file."}), 400

#         payload = run_inference(texts)
#         return jsonify(payload)
#     except Exception as exc:
#         print(f"[ERROR][app] /api/upload failed: {exc}")
#         return jsonify({"error": "CSV processing failed.", "details": str(exc)}), 500


# @app.route("/api/scrape", methods=["POST"])
# def scrape_url():
#     try:
#         data = request.get_json(force=True)
#         url = (data.get("url") or "").strip()
#         max_reviews = int(data.get("max_reviews", 50))

#         if not url:
#             return jsonify({"error": "No URL provided."}), 400

#         texts = scraper.scrape_reviews(url, max_reviews=max_reviews)
#         if not texts:
#             return jsonify({"error": "No reviews scraped from URL."}), 404

#         payload = run_inference(texts)
#         return jsonify(payload)
#     except Exception as exc:
#         print(f"[ERROR][app] /api/scrape failed: {exc}")
#         return jsonify({"error": "Scraping failed.", "details": str(exc)}), 500


# if __name__ == "__main__":
#     os.makedirs("static", exist_ok=True)
#     app.run(debug=True, host="0.0.0.0", port=5000)




import csv
import os
from io import StringIO
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request

from scraper import ReviewScraper
from utils import HybridSentimentEngine, extract_common_words, summarize_sentiments

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

engine = HybridSentimentEngine()
scraper = ReviewScraper()


def build_chart(counts: dict, filename: str = "sentiment_distribution.png") -> str:
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(label, 0) for label in labels]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=["#22c55e", "#facc15", "#ef4444"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")

    output_path = os.path.join("static", filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[DEBUG][app] Chart generated at: {output_path}")
    return f"/{output_path}"


def run_inference(texts: List[str]) -> dict:
    results = [engine.predict_single(text) for text in texts]
    counts = summarize_sentiments(results)
    common_words = extract_common_words(texts, top_n=10)
    chart_path = build_chart(counts)

    return {
        "results": results,
        "counts": counts,
        "common_words": [{"word": w, "count": c} for w, c in common_words],
        "chart": chart_path,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_text():
    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"error": "No text provided."}), 400

        payload = run_inference([text])
        return jsonify(payload)
    except Exception as exc:
        print(f"[ERROR][app] /api/predict failed: {exc}")
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request."}), 400

        file = request.files["file"]
        filename = file.filename.lower()
        if not (filename.endswith(".csv") or filename.endswith(".txt")):
            return jsonify({"error": "Only CSV or TXT files are supported."}), 400

        raw_text = file.stream.read().decode("utf-8", errors="ignore")
        texts = []

        if filename.endswith(".csv"):
            stream = StringIO(raw_text)
            reader = csv.DictReader(stream)

            for row in reader:
                # expects a column named 'text', fallback to first column
                if "text" in row and row["text"]:
                    texts.append(row["text"])
                elif row:
                    first_value = next(iter(row.values()), "")
                    if first_value:
                        texts.append(first_value)
        else:
            # TXT input: each non-empty line is treated as a separate review.
            texts = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if not texts:
            return jsonify({"error": "No usable text rows found in file."}), 400

        payload = run_inference(texts)
        return jsonify(payload)
    except Exception as exc:
        print(f"[ERROR][app] /api/upload failed: {exc}")
        return jsonify({"error": "CSV processing failed.", "details": str(exc)}), 500


@app.route("/api/scrape", methods=["POST"])
def scrape_url():
    try:
        data = request.get_json(force=True)
        url = (data.get("url") or "").strip()
        max_reviews = int(data.get("max_reviews", 200))

        if not url:
            return jsonify({"error": "No URL provided."}), 400

        texts = scraper.scrape_reviews(url, max_reviews=max_reviews)
        if not texts:
            return jsonify({"error": "No reviews scraped from URL."}), 404

        payload = run_inference(texts)
        return jsonify(payload)
    except Exception as exc:
        print(f"[ERROR][app] /api/scrape failed: {exc}")
        return jsonify({"error": "Scraping failed.", "details": str(exc)}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)