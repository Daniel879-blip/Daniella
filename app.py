# =========================================================
# Daniella AI - Friendly Smart Assistant with Memory
# Using Streamlit + OpenAI API
# =========================================================

# Import required libraries
import streamlit as st
import openai

# =========================================================
# 1. API KEY LOADING
# =========================================================
# We store the API key safely inside `.streamlit/secrets.toml`
# Example content of secrets.toml:
# [general]
# OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
#
# We try to load it here. If missing, Daniella will not work.
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    api_status = True
except KeyError:
    api_status = False

# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================
# This sets the title, emoji icon, and layout of the Streamlit app.
st.set_page_config(
    page_title="Daniella AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 3. SIDEBAR UI
# =========================================================
st.sidebar.title("🤖 Daniella AI")

# Show API key status in the sidebar
if api_status:
    st.sidebar.success("✅ API Key loaded! Daniella is online.")
else:
    st.sidebar.error("❌ API Key not found. Please add it to .streamlit/secrets.toml")

# Helpful text in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("👋 Ask me anything and I'll do my best to help!")
st.sidebar.markdown("📂 You can also upload an image for me to analyze.")

# Test connection button
if api_status and st.sidebar.button("🔌 Test Daniella Connection"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello Daniella, are you online?"}],
        )
        st.sidebar.success("Daniella says: " + response["choices"][0]["message"]["content"])
    except Exception as e:
        st.sidebar.error(f"⚠️ Connection failed: {e}")

# =========================================================
# 4. SESSION MEMORY (Chat History)
# =========================================================
# We use Streamlit's session_state to remember the chat
# between user and Daniella.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are Daniella, a friendly and smart AI assistant."}
    ]

# =========================================================
# 5. MAIN INTERFACE
# =========================================================
# Title and greeting
st.title("💬 Chat with Daniella")
st.write("Hello! I'm **Daniella**, your friendly AI assistant. Ask me anything!")

# Input box for user question
user_input = st.text_area("Type your question here:", height=100)

# File uploader for images (optional feature for future)
uploaded_file = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])

# =========================================================
# 6. GETTING ANSWER FROM DANIELLA
# =========================================================
if st.button("Ask Daniella"):
    # If API key is missing
    if not api_status:
        st.error("❌ I can't answer right now — API Key is missing.")

    # If user forgot to type a question
    elif user_input.strip() == "":
        st.warning("⚠️ Please type a question before asking me.")

    # If everything is fine
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Save user's message into the chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Send the conversation (with memory) to OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.chat_history,
                )

                # Get Daniella's reply
                answer = response["choices"][0]["message"]["content"]

                # Save Daniella's reply into chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # Display Daniella's answer
                st.success("✅ Daniella's Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# =========================================================
# 7. SHOW UPLOADED IMAGE
# =========================================================
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.info("📷 I can describe this image for you in future upgrades!")

# =========================================================
# 8. SHOW CHAT HISTORY
# =========================================================
st.markdown("## 🗂️ Conversation History")
# We skip the first item (system prompt)
for chat in st.session_state.chat_history[1:]:
    if chat["role"] == "user":
        st.markdown(f"**👤 You:** {chat['content']}")
    else:
        st.markdown(f"**🤖 Daniella:** {chat['content']}")
