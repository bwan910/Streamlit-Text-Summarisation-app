import streamlit as st
from streamlit import components
import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
#from bs4 import BeautifulSoup
import validators
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from fpdf import FPDF
import base64
import string
from nltk.translate.bleu_score import sentence_bleu
import statistics
import docx2txt
from rouge import Rouge

stop_words = stopwords.words('english')
rouge = Rouge()


def generate_summary():
    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: white;'>Upload a file or paste your text</h2>", unsafe_allow_html=True)
    
    # inputs boxes
    st.text('')
    uploaded_file = st.file_uploader('Upload Article File', type=['pdf','docx'])
    print(uploaded_file)
    st.markdown("<h2 style='text-align: center; color: white;'>OR</h2>", unsafe_allow_html=True)

    input_text = st.text_area('Paste Article Text')
    number = st.number_input('Type the summarised sentence number you need:  (If is set to 0, then will be default to 5 sentences)', step=1)

    #def create_download_link(val, filename):
     #   b64 = base64.b64encode(val)  # val looks like b'...'
    #    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

    # function for download the summary to txt file
    def export_button(data):
           st.download_button(
            label="Export Summary to a File!",
            data=data,
            file_name="Summary.txt",
            mime="application/octet-stream"
        )
        #if st.button('Export Summarize Text'):
        # pdf = FPDF()
        # pdf.add_page()
        # pdf.set_font('Arial', 'B', 11)
        # pdf.write(5,data)
        # #pdf.cell(40, 10, data)
        # #rite_pdf = create_download_link(pdf.output("Summary.pdf"))
        # html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Summary")
        # st.markdown(html, unsafe_allow_html=True)
      
    def gen_file_summary(file_data):

        sentences = sent_tokenize(file_data) # tokenize sentences
        stop_words = stopwords.words('english')
        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        #clean_sentences = dict.fromkeys(map(ord, '\n' + string.punctuation.replace('.','')))  #remove all punctuation except fullstop
        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        # function to remove stopwords
        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new
        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        # Extract word vectors
        word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        content = []    # variable for storing the summary in list

        st.title("Summarised Text")

        if number == 0:         # if no number sentence is entered, then set default as display 5 sentences of summary
            if len(sentences) < 5:      # if the length of sentence is shorter than 5 or unable to extract, then print error message
                st.error("Text too short to summarize or unable to extact text data")
            else:
                for i in range(5):
                    content.append(ranked_sentences[i][1])      #store the summary to a list
                    st.markdown('{}'.format(ranked_sentences[i][1]))  # display summary
                     # Calculating the Bleu scores for each sentence
                    reference = str(ranked_sentences[i][1]).split(".")
                    candidate = list(file_data.split("."))[i]
                    bleu_score = sentence_bleu(reference[:len(candidate)], candidate)
                    print("Bleu Score Sentence "+str(i)+": "+str(bleu_score))
                    # Calculate Rouge score
                    reference = ranked_sentences[i][1]
                    print("Rouge score Sentence "+str(i)+": "+str(rouge.get_scores(reference,file_data)))
                    print("\n")

                # Calling export button 
                convert = '\n\n'.join(map(str,content))   # since the summary is in a list convert back to string in order to allow for download
                export_button(convert)
        elif number > len(sentences):
            st.error('The chosen number of sentences for summary is too long. Text does not have enough sentences')
        else:
            for i in range(number):
                st.markdown('{}'.format(ranked_sentences[i][1]))  # display summary
                content.append(ranked_sentences[i][1])
                 # Calculating the Bleu scores for each sentence
                reference = str(ranked_sentences[i][1]).split(".")
                candidate = list(file_data.split("."))[i]
                bleu_score = sentence_bleu(reference[:len(candidate)], candidate)
                print("Bleu Score Sentence "+str(i)+": "+str(bleu_score))
                # Calculate Rouge score
                reference = ranked_sentences[i][1]
                print("Rouge score Sentence "+str(i)+": "+str(rouge.get_scores(reference,file_data)))
                print("\n")


            # Calling export button for export summary to text file
            convert = '\n\n'.join(map(str,content))
            export_button(convert)


    def file_summary():
        
        # extract docx contents
        file_path = uploaded_file.name
        #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
       # string_data = stringio.read()
        #print(stringio)
        #print(string_data)

        if file_path.endswith('.docx'):
             file_data = docx2txt.process(file_path)
             file_data = re.sub("\n", " ", file_data)   # Remove newlines
             gen_file_summary(file_data)
        
        else: 
            # Extract PDF content
            output_string = StringIO()
            with open(file_path, 'rb') as in_file:
                parser = PDFParser(in_file)
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                for page in PDFPage.create_pages(doc):
                    interpreter.process_page(page)

            file_data = output_string.getvalue()  #get file contents value
            file_data = re.sub("\n", " ", file_data)   # Remove newlines
            gen_file_summary(file_data)
           
    def inputText_summary():
        if((input_text>='a' and input_text<= 'z') or (input_text>='A' and input_text<='Z')):
            sentences = sent_tokenize(input_text)   #tokenize sentence
            sentences = list(dict.fromkeys(sentences))  # remove duplicated sentences
            
            # remove punctuations, numbers and special characters
            clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

            # make alphabets lowercase
            clean_sentences = [s.lower() for s in clean_sentences]

            # function to remove stopwords
            def remove_stopwords(sen):
                sen_new = " ".join([i for i in sen if i not in stop_words])
                return sen_new

            # remove stopwords from the sentences
            clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

            # Extract word vectors
            word_embeddings = {}
            f = open('glove.6B.100d.txt', encoding='utf-8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
            f.close()

            sentence_vectors = []
            for i in clean_sentences:
                if len(i) != 0:
                    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
                else:
                    v = np.zeros((100,))
                sentence_vectors.append(v)

            # similarity matrix
            sim_mat = np.zeros([len(sentences), len(sentences)])

            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
            content = []    # variabl for storing the summary in list

            st.title("Summarised Text")
        

            if number == 0:          # if no number sentence is entered, then set default as display 5 sentences of summary
                if len(sentences) < 5:       # if the length of sentence is shorter than 5 or unable to extract, then print error message
                    st.error("Text too short to summarize")
                else:
                    for i in range(5):
                        st.markdown('{}'.format(ranked_sentences[i][1]))  # display summary
                        content.append(ranked_sentences[i][1])
                 
                        # Calculating the Bleu scores for each sentence
                        reference = str(ranked_sentences[i][1]).split(".")
                        candidate = list(input_text.split("."))[i]
                        bleu_score = sentence_bleu(reference[:len(candidate)], candidate)
                        #st.write("BLEU Score: " +str(score))
                        #print("Bleu Score Sentence "+str(i)+": "+str(bleu_score))
                        #print("\n")

                        # Calculate Rouge score
                        reference = ranked_sentences[i][1]
                        print("Rouge score Sentence "+str(i)+": "+str(rouge.get_scores(reference,input_text)))
                        print("\n")

                     # Call export button for export to text file   
                    convert = '\n\n'.join(map(str,content))
                    export_button(convert)

            elif number > len(sentences): # for when the choosen number is longer than inputted print error msg
                st.error('The chosen number of sentences for summary is too long. Text does not have enough sentences')
            else:
                for i in range(number):
                    st.markdown('{}'.format(ranked_sentences[i][1])) 
                    content.append(ranked_sentences[i][1])   #store the summary to a list
            
                    # Calculating the Bleu scores for each sentence
                    reference = str(ranked_sentences[i][1]).split(".")
                    candidate = list(input_text.split("."))[i]
                    bleu_score = sentence_bleu(reference[:len(candidate)], candidate)
                    #st.write("BLEU Score: " +str(score))
                    print("Bleu Score Sentence "+str(i)+": "+str(bleu_score))
                    print("\n")

                        # Calculate Rouge score
                   # reference = ranked_sentences[i][1]
                   # print("Rouge score Sentence "+str(i)+": "+str(rouge.get_scores(reference,input_text)))
                   # print("\n")

                convert = '\n\n'.join(map(str,content))  # since the summary is in a list convert back to string in order to allow for download
                export_button(convert)
        else:
            st.error('Please input English Sentences only')

            

    if st.button('Summarise'):
    # when no file and no input text are given then print error message
       # if uploaded_file is None and len(input_text) <= 0 and len(input_url) <= 0:
        if uploaded_file is None and len(input_text) <= 0:
            st.error('Upload a file or enter text')
        elif uploaded_file is not None and len(input_text) > 1:
            st.error('Cannot insert two inputs')

        # when there is a file uploaded then run file_features
        elif uploaded_file is not None:
            file_summary()
        else:
            inputText_summary()
   

def main():
    # front end elements of the web page
    html_temp = """
        <div style="background-color:blue;padding:10px">
        <h1 style="color:white;text-align:center;">Text Summarization for Research</h1>
        </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)
    generate_summary()


if __name__ == '__main__':
    main()