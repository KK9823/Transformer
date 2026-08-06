# Run this script to clean raw .txt files downloaded from gutenberg.org
# Note: Most files will usually have table of contents and stuff even after the start (or before the end too)
#       You will have to get rid of those lines manually


print("Do not need to include the raw/ and .txt. Just enter the name of the file only")
filename = input("Enter filename to clean: ")

if filename.endswith(".txt"):
    filename = filename[:-4]

if filename.startswith("raw/"):
    filename = filename[4:]

with open(f"raw/{filename}.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    lines = raw_text.splitlines()

    # Identify start of the text
    i = 0
    while not lines[i].startswith("*** START"):
        i += 1

    lines = lines[i+1:] # Cut off everything before the start

    i = 0
    while not lines[i].startswith("*** END"):
        i += 1

    lines = lines[:i] # Cut off everything after the end

    with open(f"processed/{filename}_processed.txt", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lines))