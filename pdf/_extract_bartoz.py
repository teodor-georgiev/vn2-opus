import pdfplumber, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with pdfplumber.open(r'j:/Toni/Vibing/VN2_Opus/pdf/Bartoz.pdf') as pdf:
    out = []
    for i, page in enumerate(pdf.pages):
        out.append(f'=== PAGE {i+1} ===')
        out.append(page.extract_text() or '[no text]')
        out.append('')
    txt = '\n'.join(out)

with open(r'j:/Toni/Vibing/VN2_Opus/pdf/Bartoz.txt', 'w', encoding='utf-8') as f:
    f.write(txt)

print('pages:', len(pdf.pages), 'chars:', len(txt))
