import os
import PyPDF2
import random
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="rethinkl_test1")

pdf_name = input('Enter name of file: ')
pdfFileObj = open(pdf_name + '.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
txt_file = pageObj.extractText()

random_name = ''
for i in range(0,5):
    n = random.randint(1,30)
    random_name += str(n)

dir_path = os.getcwd()
dir_path = os.path.join(dir_path, 'NewData\\' + random_name + '.txt')

myText = open(dir_path, 'w+')
myText.write(txt_file)
myText.close()

doc_dir = "/NewData" 

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(dicts)
