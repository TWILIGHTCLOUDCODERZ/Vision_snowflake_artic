import streamlit as st
import replicate
import os
import pandas as pd
from transformers import AutoTokenizer
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import seaborn as sns
import speech_recognition as sr

# App title
st.set_page_config(page_title="Arctic-Vision")

def main():
    """Execution starts here."""
    get_replicate_api_token()
    display_sidebar_ui()
    init_chat_history()
    display_chat_messages()
    handle_file_upload()
    handle_speech_input()
    get_and_process_prompt()

def get_replicate_api_token():
    os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']

def display_sidebar_ui():
    with st.sidebar:
        st.subheader("Adjust model parameters")
        st.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01, key="temperature")
        st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, key="top_p")
        st.button('Clear chat history', on_click=clear_chat_history)
        st.button("Voice Input", on_click=recognize_speech)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Vision, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]
    st.session_state.chat_aborted = False

def init_chat_history():
    """Create a st.session_state.messages list to store chat messages"""
    if "messages" not in st.session_state:
        clear_chat_history()

def display_chat_messages():
    # Set assistant icon to Snowflake logo
    icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}

    # Display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text to the Model."""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

def abort_chat(error_message: str):
    """Display an error message requiring the chat to be cleared. Forces a rerun of the app."""
    assert error_message, "Error message must be provided."
    error_message = f":red[{error_message}]"
    if st.session_state.messages[-1]["role"] != "assistant":
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.session_state.messages[-1]["content"] = error_message
    st.session_state.chat_aborted = True
    st.rerun()

def get_and_process_prompt():
    """Get the user prompt and process it."""
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
            response = generate_arctic_response()
            st.write_stream(response)

    if st.session_state.chat_aborted:
        st.button('Reset chat', on_click=clear_chat_history, key="clear_chat_history")
        st.chat_input(disabled=True)
    elif prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

def generate_arctic_response():
    """String generator for the Snowflake Arctic response."""
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("user\n" + dict_message["content"] + "")
        else:
            prompt.append("assistant\n" + dict_message["content"] + "")
    
    prompt.append("assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    num_tokens = get_num_tokens(prompt_str)
    max_tokens = 1500
    
    if num_tokens >= max_tokens:
        abort_chat(f"Conversation length too long. Please keep it under {max_tokens} tokens.")
    
    st.session_state.messages.append({"role": "assistant", "content": ""})
    for event_index, event in enumerate(replicate.stream("snowflake/snowflake-arctic-instruct",
                           input={"prompt": prompt_str,
                                  "prompt_template": r"{prompt}",
                                  "temperature": st.session_state.temperature,
                                  "top_p": st.session_state.top_p,
                                  })):
        st.session_state.messages[-1]["content"] += str(event)
        yield str(event)

def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or XLSX file for analysis", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[1]
        with st.spinner("Analyzing the uploaded file..."):
            analyze_file(uploaded_file, file_type)

def analyze_file(file, file_type):
    df = None
    if file_type == "csv":
        file_content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
    elif file_type == "vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_content = file.read()
        df = pd.read_excel(BytesIO(file_content))
    
    if df is not None:
        st.write(df)
        analysis_prompt = f"Analyze the following data:\n\n{df.head(10).to_string()}\n\nProvide insights based on this data."
        st.session_state.messages.append({"role": "user", "content": analysis_prompt})
        response = generate_arctic_response()
        analysis_results = "".join([event for event in response])
        st.session_state.messages.append({"role": "assistant", "content": analysis_results})
        
        chart_paths = visualize_data(df)
        download_pdf(analysis_results, chart_paths)
    else:
        st.error("Unsupported file type. Please upload a CSV or XLSX file.")

def visualize_data(df):
    st.sidebar.header("Data Visualization")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    chart_paths = []

    if len(numeric_columns) >= 1:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Summary Statistics")
            st.write(df.describe())
        with col2:
            st.header("Correlation Matrix")
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, ax=ax)
            st.pyplot(fig)
            chart_path = "correlation_matrix.png"
            plt.savefig(chart_path)
            chart_paths.append(chart_path)

    if len(numeric_columns) >= 2:
        st.header("Pair Plot")
        fig = sns.pairplot(df[numeric_columns])
        st.pyplot(fig)
        chart_path = "pair_plot.png"
        fig.savefig(chart_path)
        chart_paths.append(chart_path)

    st.header("Distribution of Numeric Columns")
    for col in numeric_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        chart_path = f"distribution_{col}.png"
        plt.savefig(chart_path)
        chart_paths.append(chart_path)

    st.header("Pie Chart for Categorical Columns")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        fig, ax = plt.subplots()
        value_counts = df[col].value_counts()
        if len(value_counts) > 10:
            value_counts = value_counts[:10]
        value_counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        chart_path = f"pie_chart_{col}.png"
        plt.savefig(chart_path)
        chart_paths.append(chart_path)

    return chart_paths

def generate_pdf(content, chart_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, content)
    
    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=10, w=190)
    
    return pdf.output(dest="S").encode("latin1")

def download_pdf(content, chart_paths):
    pdf = generate_pdf(content, chart_paths)
    b64 = base64.b64encode(pdf).decode('latin1')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="results.pdf">Download PDF</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.session_state.messages.append({"role": "user", "content": text})
        with st.chat_message("user", avatar="⛷️"):
            st.write(text)
        st.experimental_rerun()
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

def handle_speech_input():
    if "speech_input" not in st.session_state:
        st.session_state.speech_input = ""

if __name__ == "__main__":
    main()
