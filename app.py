from flask import Flask,render_template,request,jsonify
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import requests
import os 


from urllib.parse import urlparse,parse_qs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#setup
load_dotenv()

app = Flask(__name__)

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    url = str(url)
    parsed_url = urlparse(url)

    # Case 1: normal youtube.com/watch?v=
    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]

    # Case 2: youtu.be short link
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")

    # Case 3: youtube shorts
    if "shorts" in parsed_url.path:
        return parsed_url.path.split("/")[-1]

    return None
def extract_transcript(youtube_url):
    video_id = extract_video_id(youtube_url)

    if not video_id:
        return None

    # 🔁 Try direct transcript first (fastest)
    for i in range(2):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

            text = " ".join([item["text"] for item in transcript])
            return text

        except Exception as e:
            print(f"Direct attempt {i+1} failed:", e)

    # 🔁 Fallback: auto-generated transcripts
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # try English generated
        try:
            transcript = transcript_list.find_generated_transcript(['en']).fetch()

        except:
            # last fallback: take any available transcript
            transcript = next(iter(transcript_list)).fetch()

        text = " ".join([item.text for item in transcript])
        return text

    except Exception as e2:
        print("Fallback failed:", e2)
        return None#chunks
def chunk_text(text,chunk_size=500):
    words = text.split()

    chunks=[]

    for i in range(0,len(words),chunk_size):
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks



    
def search(question, chunks):
    docs = chunks + [question]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(docs)

    question_vector = vectors[-1]
    chunk_vectors = vectors[:-1]

    scores = cosine_similarity(question_vector, chunk_vectors)[0]

    top_indices = scores.argsort()[-3:][::-1]

    return [chunks[i] for i in top_indices]

def Openrouter(question,context):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        return {"error":"API Key is missing"}

    url =  "https://openrouter.ai/api/v1/chat/completions"

    prompt = f"""
You are a highly accurate YouTube video understanding assistant.

Your job is to answer the user's question ONLY using the provided transcript context.

---

RULES:

1. Use ONLY the given context. Do NOT guess or assume external knowledge.
2. If the context is unclear, noisy, or contains only filler like:
   [music], [applause], [laughter], repetition, or meaningless words,
   then clearly say:
   "The video does not contain enough clear information to answer this question."

3. If the video is a music/video with no factual content, respond:
   "This appears to be a music or non-informational video, so the answer cannot be determined from the transcript."

4. If the answer exists, give a clear, short, and direct response.

5. Do NOT mention embeddings, chunks, or system logic.

6. Do NOT hallucinate song titles, names, or facts not present in the context.

---

QUESTION:
{question}

---

TRANSCRIPT CONTEXT:
{context}

---

FINAL ANSWER:
"""
    headers = {
            "Authorization":f"Bearer {OPENROUTER_API_KEY}",
            "content-type":"application/json"
        }
    payload ={
            "model" : "openai/gpt-4o-mini",
            "messages":[
                {'role':'user',"content":prompt}
            ]
        }
    response = requests.post(url,headers=headers,json=payload)
    print("STATUS:", response.status_code)
    print("TEXT:", response.text)

    result = response.json()
    print('OPENROUTER_RESPONSE:',result)

    if "choices" not in result:
        return {"error":result}

    return result["choices"][0]["message"]["content"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/youtube',methods=["POST"])
def youtube():
    youtube_url = request.form.get("video_url")
    question = request.form.get("question")
    print("URL:", youtube_url)
    print("QUESTION:", question)

    text=extract_transcript(youtube_url)
    if not text:
        return jsonify({
            "error": "Transcript could not be fetched for this video."
        }), 400
    print("TRANSCRIPT LENGTH:", len(text))

    chunks = chunk_text(text)
    print("NUMBER OF CHUNKS:", len(chunks))

    

    top_chunks = search(question,chunks)

    print("TOP CHUNK:")
    print(top_chunks[0])

    context = '\n'.join(top_chunks)

    result = Openrouter(question,context)
    print("RESULT:", result)

    return jsonify({
        "question": question,
        "answer": result,
        "context": top_chunks
    })

if __name__ =="__main__":
    app.run(host="0.0.0.0", port=10000, debug=True, use_reloader=False)
