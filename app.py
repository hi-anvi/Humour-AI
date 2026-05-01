from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ai_model import generate_joke

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/joke", methods=["POST"])
def joke():
    data = request.get_json()
    situation = data.get("situation", "").strip()
    mode = data.get("mode", "one-liner").strip()

    if not situation:
        return jsonify({"error": "No situation provided"}), 400

    result = generate_joke(situation, mode)

    if result["success"]:
        return jsonify({"joke": result["joke"]})
    else:
        return jsonify({"error": result["error"]}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
