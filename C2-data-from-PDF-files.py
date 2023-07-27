import PyPDF4
from pathlib import Path

folder = "./"

def print_metadata(pdf_reader):
    # print the meta data
    metadata = pdf_reader.getDocumentInfo()
    print (f"Title: {metadata.title}")
    print (f"Author: {metadata.author}")
    print (f"Subject: {metadata.subject}")

def save_content(pdf_reader):
    # print number of pages in pdf file
    pages = pdf_reader.numPages
    print(f"Pages: {pages}")

    # get content for each page
    page = 0 
    while page < pages:
        pdf_page = pdf_reader.getPage(page)
        print(pdf_page.extractText())
        page+=1
        # write each page to a database here

pathlist = Path(folder).rglob('*.pdf')
for file_name in pathlist:
    # because path is object not string
    file_name = str(file_name)

    pdf_file = open(file_name,'rb')
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)
    print (f"File name: {file_name}")
    print_metadata(pdf_reader)
    save_content(pdf_reader)

    # tidy up
    pdf_file.close()
