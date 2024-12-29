import fitz
import copy
import pandas as pd
import re
import tiktoken
import json

tokenizer = tiktoken.get_encoding("gpt2")

def get_toc(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()
    return toc
	
def remove_headers_footers(page, top_margin=50, bottom_margin=50):
        rect = page.rect
        cropped_rect = fitz.Rect(rect.x0, rect.y0 + top_margin, rect.x1, rect.y1 - bottom_margin)
        text = page.get_text(clip=cropped_rect)

        return text

def extract_text_between_pages(doc, start_page, end_page=None, ):
    text = ""

    if end_page is None:
        text += remove_headers_footers(doc.load_page(start_page - 1))
    else:
        for page_num in range(start_page - 1, end_page):
            page = doc.load_page(page_num)
            text += remove_headers_footers(page) + "\n"

    return text

def add_to_structure(toc_structure, item, level):
    if len(toc_structure) == 0 or level == 1:
        toc_structure.append(item)
    else:
        parent = toc_structure[-1]
        if 'children' not in parent:
            parent['children'] = []

        if parent['level'] < level - 1:
            add_to_structure(parent['children'], item, level)
        else:
            parent['children'].append(item)

def count_words_and_tokens(text):
    words = len(text.split())
    tokens = len(tokenizer.encode(text))
    return words, tokens

def get_content_between_headings(text, current_heading, next_heading):
    current_heading_match = re.search(re.escape(current_heading), text, re.IGNORECASE)
 
    if next_heading:
        next_heading_matches = re.findall(re.escape(next_heading), text, re.IGNORECASE)
    else:
        next_heading_matches = []
       
    if current_heading_match:
        start_index = current_heading_match.start()
    else:
        start_index = 0

    if next_heading_matches:
        last_next_heading = next_heading_matches[-1]
        next_heading_match = re.search(re.escape(last_next_heading), text, re.IGNORECASE)
        end_index = next_heading_match.start()
    else:
        end_index = len(text)
       
    return text[start_index:end_index]

def doc_parser(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    toc = copy.deepcopy(toc)

    last_page_number = int(doc.page_count)
    nested_toc = []
    toc_rows = []
    header_id = 1
    parent_stack = []

    for i, item in enumerate(toc):
        level, title, start_page = item
        while parent_stack and parent_stack[-1]['level'] >= level:
            parent_stack.pop()

        parent_titles = [p['title'] for p in parent_stack]

        if i + 1 < len(toc):
            next_item = toc[i + 1]
            _, next_title, end_page = next_item
        else:
            next_title = None
            end_page = None

        text = extract_text_between_pages(doc, start_page, end_page)
        refined_text = get_content_between_headings(text, title, next_title)
        word_count, token_count = count_words_and_tokens(refined_text)

        current_item = {
            'header_id': header_id,
            'title': title,
            'level': level,
            'start_page': start_page,
            'end_page': last_page_number if end_page is None else end_page,
            'content': refined_text,
            'word_count': word_count,
            'token_count': token_count,
            'parent_titles': parent_titles,
            'children': []
        }

        add_to_structure(nested_toc, current_item, level)

        toc_rows.append({
            'header_id': header_id,  
            'title': title,
            'level': level,
            'start_page': start_page,
            'end_page': last_page_number if end_page is None else end_page,
            'content': refined_text,
            'word_count': word_count,
            'token_count': token_count,
            'parent_titles': parent_titles
        })

        parent_stack.append(current_item)

        header_id += 1

    doc.close()

    toc_df = pd.DataFrame(toc_rows)

    return nested_toc, toc_df
