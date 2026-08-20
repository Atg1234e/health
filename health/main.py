import re
import time
import random

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ----------------------------------------------------------------------
# 1. FIRST-AID KNOWLEDGE BASE
# ----------------------------------------------------------------------
FIRST_AID = {
    "cut": {
        "label": "Cut / Wound",
        "emoji": "🩹",
        "steps": [
            "Wash your hands before touching the wound if possible.",
            "Apply firm, direct pressure with a clean cloth or gauze to stop bleeding.",
            "Rinse the wound gently with clean running water to remove dirt.",
            "Apply an antiseptic and cover with a sterile bandage.",
            "Change the dressing daily and keep the area clean and dry.",
        ],
        "avoid": [
            "Don't use cotton wool directly on an open wound (fibres stick).",
            "Don't remove a deeply embedded object — stabilize it and seek help.",
        ],
        "emergency": "Seek immediate help if bleeding doesn't stop after 10 minutes of pressure, the cut is deep/gaping, or it was caused by a rusty/dirty object (tetanus risk).",
    },
    "burn": {
        "label": "Burn",
        "emoji": "🔥",
        "steps": [
            "Cool the burn under cool (not ice-cold) running water for 10–20 minutes.",
            "Remove any tight clothing or jewellery near the area before it swells.",
            "Cover loosely with a clean, non-fluffy cloth or cling film.",
            "Take a paracetamol-type pain reliever if needed and available to you.",
        ],
        "avoid": [
            "Don't apply ice, butter, toothpaste, or oil to a burn.",
            "Don't burst any blisters that form.",
        ],
        "emergency": "Call for help immediately for large burns, burns on the face/hands/genitals, chemical or electrical burns, or if the skin is charred/white.",
    },
    "snake_bite": {
        "label": "Snake Bite",
        "emoji": "🐍",
        "steps": [
            "Keep the person calm and completely still — movement spreads venom faster.",
            "Keep the bitten limb below heart level and immobilize it like a fracture.",
            "Remove rings, watches, or tight clothing near the bite before swelling starts.",
            "Note the time of the bite and, if safe, the snake's appearance (don't chase it).",
            "Get to a hospital immediately — antivenom is the only real treatment.",
        ],
        "avoid": [
            "Don't cut the wound, try to suck out venom, or apply a tight tourniquet.",
            "Don't apply ice or give the person alcohol/caffeine.",
        ],
        "emergency": "All snake bites are medical emergencies — call 108 and get to a hospital right away, even if symptoms seem mild at first.",
    },
    "insect_sting": {
        "label": "Wasp / Bee Sting",
        "emoji": "🐝",
        "steps": [
            "If it's a bee, scrape out the stinger sideways with a card (don't pinch/tweeze it).",
            "Wash the area with soap and water.",
            "Apply a cold compress to reduce swelling and pain.",
            "An antihistamine or calamine lotion can ease itching/swelling if available.",
        ],
        "avoid": [
            "Don't squeeze the stinger with fingers or tweezers — it can inject more venom.",
            "Don't scratch the area, as it raises infection risk.",
        ],
        "emergency": "Call for emergency help immediately if there's difficulty breathing, swelling of the face/throat, dizziness, or hives spreading over the body — this can be anaphylaxis.",
    },
    "dog_bite": {
        "label": "Dog / Animal Bite",
        "emoji": "🐕",
        "steps": [
            "Wash the wound thoroughly with soap and running water for at least 15 minutes.",
            "Apply an antiseptic and cover with a clean bandage.",
            "Go to a doctor promptly to assess the need for a rabies vaccine course and tetanus shot.",
            "Try to note the animal (owned/stray, vaccinated if known) for the doctor.",
        ],
        "avoid": [
            "Don't ignore even a minor-looking bite — rabies risk isn't about wound size.",
            "Don't apply chili powder, turmeric, or other home remedies to the wound.",
        ],
        "emergency": "Seek medical care the same day for any animal bite that breaks the skin. Go immediately if bleeding is heavy or the bite is on the face/neck.",
    },
    "fracture": {
        "label": "Suspected Fracture",
        "emoji": "🦴",
        "steps": [
            "Keep the injured area still — avoid moving or straightening it.",
            "Support and immobilize it with a splint, rolled cloth, or by strapping to a nearby body part.",
            "Apply a cold pack wrapped in cloth to reduce swelling.",
            "Get the person to a hospital for an X-ray as soon as possible.",
        ],
        "avoid": [
            "Don't try to push a protruding bone back in.",
            "Don't give food or water if surgery might be needed soon.",
        ],
        "emergency": "Call 108 for suspected spine, neck, hip, or thigh fractures, or if the limb looks deformed or bone is visible — don't move the person.",
    },
    "nosebleed": {
        "label": "Nosebleed",
        "emoji": "🩸",
        "steps": [
            "Sit down and lean slightly forward (not back).",
            "Pinch the soft part of the nose firmly for 10–15 minutes without releasing.",
            "Breathe through your mouth while pinching.",
            "Apply a cold compress to the bridge of the nose.",
        ],
        "avoid": [
            "Don't tilt the head backward — blood can run down the throat.",
            "Don't pack the nose with tissue and forget about it.",
        ],
        "emergency": "Seek help if bleeding continues beyond 20–30 minutes, is very heavy, or follows a head injury.",
    },
    "choking": {
        "label": "Choking",
        "emoji": "🫁",
        "steps": [
            "Encourage the person to cough forcefully if they still can.",
            "If they can't cough/speak/breathe, give up to 5 sharp back blows between the shoulder blades.",
            "If that fails, give up to 5 abdominal thrusts (Heimlich manoeuvre).",
            "Alternate back blows and abdominal thrusts until the object is dislodged or help arrives.",
        ],
        "avoid": [
            "Don't perform abdominal thrusts on infants under 1 year — use back blows and chest thrusts instead.",
            "Don't blindly sweep the mouth with fingers unless you can clearly see the object.",
        ],
        "emergency": "Call 108 immediately if the person becomes unconscious or the obstruction doesn't clear — begin CPR if trained and the person stops responding.",
    },
    "electric_shock": {
        "label": "Electric Shock",
        "emoji": "⚡",
        "steps": [
            "Turn off the power source first, or use a dry, non-conductive object (wooden stick) to separate the person from the source.",
            "Once safe, check if the person is breathing and responsive.",
            "Cover any burns with a clean, dry cloth.",
            "Keep the person warm and still while waiting for help.",
        ],
        "avoid": [
            "Never touch the person directly while they're still in contact with the electrical source.",
            "Don't use anything metal or wet to move the person or the source.",
        ],
        "emergency": "Always call 108 after any significant electric shock, even if the person seems fine — internal effects can appear later.",
    },
    "fainting": {
        "label": "Fainting",
        "emoji": "💫",
        "steps": [
            "Lay the person down flat and raise their legs about 12 inches to improve blood flow to the brain.",
            "Loosen any tight clothing around the neck and waist.",
            "Ensure fresh air — open a window or fan them.",
            "As they come round, keep them lying down for a few minutes before sitting up slowly.",
        ],
        "avoid": [
            "Don't give food or water until they're fully alert.",
            "Don't splash cold water on the face or slap them.",
        ],
        "emergency": "Call for help if the person doesn't regain consciousness within a minute, has an injury from falling, or faints repeatedly.",
    },
    "heatstroke": {
        "label": "Heatstroke / Heat Exhaustion",
        "emoji": "🥵",
        "steps": [
            "Move the person to a cool, shaded place immediately.",
            "Remove excess clothing and cool them with damp cloths or a fan.",
            "Give sips of cool water only if they are fully conscious and alert.",
            "Apply cold packs to the neck, armpits, and groin to bring temperature down fast.",
        ],
        "avoid": [
            "Don't give water if the person is confused, vomiting, or losing consciousness.",
            "Don't use ice-cold water for a sudden dunk — cool gradually.",
        ],
        "emergency": "Call 108 immediately if body temperature is very high, the person is confused, has stopped sweating, or loses consciousness — heatstroke can be fatal.",
    },
    "poisoning": {
        "label": "Poisoning / Ingestion",
        "emoji": "☠️",
        "steps": [
            "Try to identify what was ingested and how much — keep the container/label if possible.",
            "Call 108 or a poison control helpline right away and follow their instructions.",
            "If the person is unconscious, place them in the recovery position and watch their breathing.",
            "Keep a sample of vomit if it occurs, as it may help doctors identify the poison.",
        ],
        "avoid": [
            "Don't induce vomiting unless a medical professional specifically tells you to.",
            "Don't give milk, water, or any home remedy unless instructed by a professional.",
        ],
        "emergency": "All suspected poisoning cases need urgent medical attention — call 108 immediately.",
    },
    "fever": {
        "label": "Fever",
        "emoji": "🌡️",
        "steps": [
            "Rest and drink plenty of fluids — water, ORS, or clear soups.",
            "Dress in light clothing and keep the room comfortably cool.",
            "Take paracetamol as per standard dosing if needed to bring the fever down.",
            "Use a lukewarm sponge on the forehead/body if the fever feels very high.",
        ],
        "avoid": [
            "Don't wrap the person in heavy blankets or clothing.",
            "Don't use alcohol rubs or ice baths to cool the body.",
        ],
        "emergency": "Seek medical care urgently for fever above 103°F (39.4°C), fever with a stiff neck/rash/severe headache, in infants under 3 months, or if it lasts more than 3 days.",
    },
    "headache": {
        "label": "Headache / Migraine",
        "emoji": "🤕",
        "steps": [
            "Rest in a quiet, dark, cool room.",
            "Stay hydrated — dehydration is a common trigger.",
            "Apply a cold or warm compress to the head or neck, whichever feels better.",
            "A mild pain reliever can help if the headache is not severe or unusual for you.",
        ],
        "avoid": [
            "Don't stare at bright screens during an attack.",
            "Don't skip meals — low blood sugar can worsen headaches.",
        ],
        "emergency": "Seek urgent care for a sudden 'worst-ever' headache, headache with vision loss/confusion/weakness, or after a head injury.",
    },
    "allergic_reaction": {
        "label": "Allergic Reaction",
        "emoji": "🤧",
        "steps": [
            "Remove or move away from the trigger (food, plant, product) if known.",
            "Take an antihistamine for mild reactions like hives or itching.",
            "Apply a cool compress to itchy or swollen skin.",
            "If the person has a prescribed epinephrine auto-injector for severe allergies, help them use it.",
        ],
        "avoid": [
            "Don't wait and watch if breathing becomes difficult — act immediately.",
            "Don't give anything by mouth if the throat feels like it's closing up.",
        ],
        "emergency": "Call 108 immediately for swelling of the face/lips/throat, difficulty breathing, or dizziness — this can be anaphylaxis, a life-threatening emergency.",
    },
}


# ----------------------------------------------------------------------
# 2. TRAINING DATA FOR THE SKLEARN CLASSIFIER
# ----------------------------------------------------------------------
TRAINING_PHRASES = {
    "cut": [
        "I cut my finger with a knife",
        "there is a deep cut on my hand and it is bleeding",
        "I got a small cut while cooking",
        "glass cut my leg and it won't stop bleeding",
        "I have a wound from a sharp object",
        "sliced my finger while chopping vegetables",
        "bleeding cut on my arm",
        "scraped my knee and it's bleeding",
    ],
    "burn": [
        "I burned my hand on the stove",
        "hot water spilled on my arm and it is burning",
        "I touched a hot pan and got burned",
        "steam burn on my hand",
        "my skin is red and blistered from a burn",
        "I got a burn from an iron",
        "fire burned my finger",
        "sunburn all over my back",
    ],
    "snake_bite": [
        "a snake bit me on the leg",
        "I was bitten by a snake in the garden",
        "snake bite on my foot with swelling",
        "cobra bit my hand",
        "I think a venomous snake bit me",
        "two puncture marks from a snake bite",
    ],
    "insect_sting": [
        "a wasp stung me",
        "bee sting on my arm swelling up",
        "I got stung by a bee and it hurts",
        "wasp bite on my neck",
        "insect sting is red and itchy",
        "hornet stung my leg",
    ],
    "dog_bite": [
        "a dog bit my leg",
        "stray dog bit me on the hand",
        "my pet dog bit my finger",
        "dog bite wound bleeding",
        "cat scratched and bit me",
        "animal bite on my arm",
    ],
    "fracture": [
        "I think I broke my arm",
        "my leg looks deformed after falling, might be a fracture",
        "severe pain and swelling in my wrist after a fall",
        "I can't move my ankle, maybe fractured",
        "bone might be broken after the accident",
        "suspected fracture in my hand",
    ],
    "nosebleed": [
        "my nose is bleeding",
        "I have a sudden nosebleed",
        "blood is coming from my nose",
        "nosebleed after blowing my nose",
        "nose won't stop bleeding",
    ],
    "choking": [
        "someone is choking on food",
        "my child is choking and can't breathe",
        "food is stuck in my throat and I can't breathe",
        "he is choking and turning blue",
        "person choking, cannot speak or cough",
    ],
    "electric_shock": [
        "I got an electric shock from a wire",
        "he touched a live wire and got shocked",
        "electric shock while using an appliance",
        "someone got electrocuted",
        "shocked by faulty switch",
    ],
    "fainting": [
        "I feel dizzy and about to faint",
        "she fainted suddenly",
        "he lost consciousness for a moment and fell",
        "feeling lightheaded and blacked out",
        "someone fainted in the heat",
    ],
    "heatstroke": [
        "I feel extremely hot and dizzy after being in the sun",
        "heatstroke symptoms, very high body temperature",
        "he collapsed from heat exhaustion",
        "feeling faint and overheated after exercise in the sun",
        "not sweating and very hot, confused",
    ],
    "poisoning": [
        "I accidentally swallowed some chemical",
        "my child ate something poisonous",
        "food poisoning with vomiting",
        "swallowed cleaning liquid by mistake",
        "possible poisoning after eating something bad",
    ],
    "fever": [
        "I have a high fever",
        "body temperature is very high with chills",
        "running a fever since morning",
        "feeling feverish and weak",
        "my child has a fever of 102",
    ],
    "headache": [
        "I have a severe headache",
        "throbbing pain in my head, feels like a migraine",
        "constant headache since yesterday",
        "my head is pounding",
        "migraine attack with light sensitivity",
    ],
    "allergic_reaction": [
        "I broke out in hives after eating peanuts",
        "my face is swelling after a bee sting, allergic reaction",
        "itchy rash all over my body",
        "throat feels tight after eating shellfish, allergic",
        "skin allergy with swelling and itching",
    ],
}

HEALTH_TIPS = [
    "Drink at least 8 glasses of water a day to stay properly hydrated.",
    "Aim for 7–9 hours of quality sleep every night — it's when your body repairs itself.",
    "Try to get at least 30 minutes of moderate physical activity most days of the week.",
    "Fill half your plate with vegetables and fruits at each meal.",
    "Take short breaks to stretch if you sit for long periods at work or study.",
    "Practice deep breathing or meditation for a few minutes daily to manage stress.",
    "Limit added sugar and ultra-processed foods where you can.",
    "Wash your hands regularly to prevent infections.",
    "Keep a basic first-aid kit at home and in your car.",
    "Get regular health check-ups even when you feel fine — prevention beats cure.",
    "Reduce screen time before bed for better sleep quality.",
    "Maintain good posture, especially if you work at a desk all day.",
    "Spend time outdoors — sunlight and fresh air support both mood and vitamin D levels.",
    "Nurture your relationships — social connection is linked to better long-term health.",
    "Don't skip breakfast; it helps stabilize energy and blood sugar through the day.",
]


# ----------------------------------------------------------------------
# 3. MODEL TRAINING (scikit-learn only)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training first-aid classifier...")
def train_classifier():
    rows = []
    for label, phrases in TRAINING_PHRASES.items():
        for p in phrases:
            rows.append({"text": p, "label": label})
    df = pd.DataFrame(rows)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(df["text"], df["label"])

    try:
        cv_scores = cross_val_score(pipeline, df["text"], df["label"], cv=3)
        cv_accuracy = float(np.mean(cv_scores))
    except Exception:
        cv_accuracy = None

    return pipeline, sorted(df["label"].unique()), cv_accuracy, len(df)


def predict_category(user_text, sklearn_pipeline):
    """Return (label, confidence) for a user's free-text query using scikit-learn."""
    probs = sklearn_pipeline.predict_proba([user_text])[0]
    top_idx = int(np.argmax(probs))
    top_label = sklearn_pipeline.classes_[top_idx]
    top_conf = float(probs[top_idx])
    return top_label, top_conf


# ----------------------------------------------------------------------
# 4. STREAMLIT APP & CUSTOM STYLING WITH BACKGROUND IMAGES
# ----------------------------------------------------------------------
st.set_page_config(page_title="First Aid & Wellness Assistant", page_icon="🩺", layout="centered")

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# Map specific background images from Unsplash based on active page
BACKGROUND_IMAGES = {
    "login": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1920&q=80",
    "First Aid Assistant": "https://images.unsplash.com/photo-1584515933487-779824d29309?auto=format&fit=crop&w=1920&q=80",
    "Healthy Lifestyle Tips": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=1920&q=80",
    "Breathing Exercise": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=1920&q=80",
    "Emergency Call": "https://images.unsplash.com/photo-1587745416684-47953f16f02f?auto=format&fit=crop&w=1920&q=80",
}

def inject_dynamic_css(bg_url):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(14, 17, 23, 0.75), rgba(14, 17, 23, 0.85)),
                            url("{bg_url}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            .main .block-container {{
                background: rgba(22, 27, 34, 0.65);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin-top: 20px;
                margin-bottom: 20px;
            }}

            @keyframes pulse-btn {{
                0% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.7); }}
                70% {{ transform: scale(1.02); box-shadow: 0 0 0 15px rgba(211, 47, 47, 0); }}
                100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(211, 47, 47, 0); }}
            }}
            
            .pulse-call-btn {{
                display: block;
                text-align: center;
                background: linear-gradient(135deg, #d32f2f, #b71c1c);
                color: white !important;
                padding: 20px;
                border-radius: 12px;
                font-size: 24px;
                font-weight: bold;
                text-decoration: none;
                margin: 20px 0;
                animation: pulse-btn 2s infinite;
                transition: all 0.3s ease;
            }}
            .pulse-call-btn:hover {{
                opacity: 0.95;
                color: white !important;
            }}

            .breath-box {{
                width: 180px;
                height: 180px;
                margin: 20px auto;
                border-radius: 50%;
                background: radial-gradient(circle, #4facfe 0%, #00f2fe 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 22px;
                box-shadow: 0 0 25px rgba(79, 172, 254, 0.6);
                transition: transform 0.8s ease-in-out, background 0.5s ease;
            }}

            .step-card {{
                background-color: rgba(255, 255, 255, 0.08);
                border-left: 4px solid #4facfe;
                padding: 12px 16px;
                margin-bottom: 8px;
                border-radius: 0 8px 8px 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- LOGIN GATE ----
if not st.session_state.logged_in:
    inject_dynamic_css(BACKGROUND_IMAGES["login"])
    st.title("🩺 First Aid & Wellness Assistant")
    st.write("Please sign in with your email to continue.")
    with st.form("login_form"):
        email_input = st.text_input("Email address")
        submitted = st.form_submit_button("Login")
        if submitted:
            if EMAIL_REGEX.match(email_input.strip()):
                st.session_state.logged_in = True
                st.session_state.user_email = email_input.strip()
                st.rerun()
            else:
                st.error("Please enter a valid email address.")
    st.caption("Note: this is a session-only login for demo purposes — no password, no data stored on a server.")
    st.stop()

# ---- MAIN APP (post-login) ----
pipeline_model, labels, cv_accuracy, n_examples = train_classifier()

st.sidebar.title("🩺 Menu")
st.sidebar.write(f"Signed in as **{st.session_state.user_email}**")
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

page = st.sidebar.radio(
    "Go to",
    ["First Aid Assistant", "Healthy Lifestyle Tips", "Breathing Exercise", "Emergency Call"],
)

# Apply dynamic page-specific background image
inject_dynamic_css(BACKGROUND_IMAGES.get(page, BACKGROUND_IMAGES["First Aid Assistant"]))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📞 **Emergency? Call 108**\n\n"
    "[Tap to call 108](tel:108)"
)

st.title("🩺 First Aid & Wellness Assistant")
st.caption(
    "⚠️ Educational tool only — not a substitute for professional medical advice. "
    "In any emergency, call 108 immediately."
)

# ---- PAGE: FIRST AID ASSISTANT ----
if page == "First Aid Assistant":
    st.header("Describe what happened")
    st.write("Type your symptom or injury in your own words, e.g. *\"a wasp stung my arm and it's swelling\"*.")

    user_text = st.text_input("What's going on?", placeholder="e.g. I cut my finger while cooking")

    if st.button("Get First Aid Guidance") and user_text.strip():
        with st.spinner("Analyzing emergency condition..."):
            time.sleep(0.3)
            label, confidence = predict_category(user_text, pipeline_model)
            info = FIRST_AID[label]

        st.success(f"{info['emoji']} Likely category: **{info['label']}**")
        st.caption(f"Model Confidence Threshold:")
        st.progress(confidence)

        st.subheader("Steps to take")
        for i, step in enumerate(info["steps"], start=1):
            st.markdown(f"<div class='step-card'><b>Step {i}:</b> {step}</div>", unsafe_allow_html=True)

        st.subheader("⚠️ Avoid")
        for d in info["avoid"]:
            st.markdown(f"- {d}")

        st.error(f"🚨 **When to call for help:** {info['emergency']}")
        st.markdown('<a href="tel:108" class="pulse-call-btn">📞 CALL 108 NOW</a>', unsafe_allow_html=True)

    with st.expander("ℹ️ About this classifier"):
        st.write(
            f"Trained on {n_examples} example phrases across {len(labels)} categories using "
            f"scikit-learn's TF-IDF vectorizer and Logistic Regression."
        )
        if cv_accuracy is not None:
            st.write(f"Cross-validated accuracy on the training set: **{cv_accuracy:.0%}** (small demo dataset).")

    st.markdown("---")
    st.subheader("Browse all first-aid topics")
    chosen = st.selectbox("Select a topic", [FIRST_AID[l]["label"] for l in labels])
    chosen_key = [l for l in labels if FIRST_AID[l]["label"] == chosen][0]
    info = FIRST_AID[chosen_key]
    st.markdown(f"### {info['emoji']} {info['label']}")
    
    tabs = st.tabs(["📋 Recommended Steps", "⛔ What NOT to do", "🚨 Emergency Criteria"])
    with tabs[0]:
        for i, step in enumerate(info["steps"], start=1):
            st.markdown(f"{i}. {step}")
    with tabs[1]:
        for d in info["avoid"]:
            st.markdown(f"- {d}")
    with tabs[2]:
        st.warning(info["emergency"])

# ---- PAGE: HEALTHY LIFESTYLE TIPS ----
elif page == "Healthy Lifestyle Tips":
    st.header("🌱 Healthy Lifestyle Tips")
    st.write("Small, consistent habits make the biggest difference over time.")

    if "tip_index" not in st.session_state:
        st.session_state.tip_index = random.randint(0, len(HEALTH_TIPS) - 1)

    st.info(f"💡 {HEALTH_TIPS[st.session_state.tip_index]}")
    if st.button("🎲 Show another tip"):
        st.session_state.tip_index = random.randint(0, len(HEALTH_TIPS) - 1)
        st.toast("Updated tip selection!", icon="✨")
        st.rerun()

    st.subheader("All tips")
    for tip in HEALTH_TIPS:
        st.markdown(f"- {tip}")

# ---- PAGE: BREATHING EXERCISE ----
elif page == "Breathing Exercise":
    st.header("🌬️ Guided Box Breathing")
    st.write(
        "Box breathing (inhale–hold–exhale–hold, 4 seconds each) is a simple technique "
        "used to calm the nervous system and reduce stress or panic."
    )

    cycles = st.slider("Number of cycles", min_value=1, max_value=8, value=4)
    seconds_per_phase = st.slider("Seconds per phase", min_value=2, max_value=8, value=4)

    start = st.button("Start breathing exercise")

    if start:
        placeholder = st.empty()
        progress = st.progress(0)
        phases = [
            ("Inhale 🫁⬆️", seconds_per_phase, 1.45, "radial-gradient(circle, #4facfe 0%, #00f2fe 100%)"),
            ("Hold ✋", seconds_per_phase, 1.45, "radial-gradient(circle, #f6d365 0%, #fda085 100%)"),
            ("Exhale 🫁⬇️", seconds_per_phase, 1.0, "radial-gradient(circle, #43e97b 0%, #38f9d7 100%)"),
            ("Hold ✋", seconds_per_phase, 1.0, "radial-gradient(circle, #fa709a 0%, #fee140 100%)"),
        ]
        total_steps = cycles * len(phases) * seconds_per_phase
        step_counter = 0

        for cycle in range(1, cycles + 1):
            for phase_name, duration, scale, bg_gradient in phases:
                for remaining in range(duration, 0, -1):
                    placeholder.markdown(
                        f"""
                        <div style="text-align: center;">
                            <p style="font-size: 18px; margin-bottom: 5px;"><b>Cycle {cycle} of {cycles}</b></p>
                            <div class="breath-box" style="transform: scale({scale}); background: {bg_gradient};">
                                {phase_name}<br>{remaining}s
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    step_counter += 1
                    progress.progress(min(step_counter / total_steps, 1.0))
                    time.sleep(1)

        placeholder.markdown("<h3 style='text-align: center;'>✅ Done! Notice how you feel.</h3>", unsafe_allow_html=True)
        st.balloons()
        st.success("Great job. You can repeat this any time you feel stressed, anxious, or overwhelmed.")

# ---- PAGE: EMERGENCY CALL ----
elif page == "Emergency Call":
    st.header("🚨 Emergency Ambulance")
    st.write("In India, **108** is the free, toll-free ambulance and emergency response number.")

    st.markdown('<a href="tel:108" class="pulse-call-btn">📞 CALL 108 NOW</a>', unsafe_allow_html=True)

    st.write("When you call, try to clearly state:")
    st.markdown(
        "- Your exact location (landmark if address is unclear)\n"
        "- What happened and how many people are affected\n"
        "- The condition of the person (conscious/breathing/bleeding, etc.)\n"
        "- A callback number"
    )

    st.subheader("Other useful numbers (India)")
    st.markdown(
        "- **100** — Police\n"
        "- **101** — Fire\n"
        "- **108** — Ambulance / Emergency Medical Services\n"
        "- **1091** — Women's Helpline\n"
        "- **1098** — Child Helpline"
    )

st.markdown("---")
st.caption(
    "This app provides general first-aid education only and is not a medical device. "
    "Always call 108 or your local emergency number for serious injuries or medical emergencies."
)
