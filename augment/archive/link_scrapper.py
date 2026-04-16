file_path = "urls.txt"
with open(file_path, "r") as file:
    urls = file.readlines()

for url in urls:
    print(url)