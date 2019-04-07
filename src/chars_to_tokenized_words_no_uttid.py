import sys

from textutils import form_tokenized_words

input_file = sys.argv[1]
output_file = sys.argv[2]

output_lines = []
with open(input_file, "r") as fh:
    for line in fh:
        utt = line.strip()
        tokenized_utt = ' '.join(form_tokenized_words(utt.split(), with_spaces=True))
        output_lines.append(("%s\n" % (tokenized_utt)))

with open(output_file, "w") as fh:
    for line in output_lines:
        fh.write(line)
