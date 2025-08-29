# 🧪 Awful Book Sanitizer: Transforming Chaos into Clarity  

> *A Rust program that leverages OpenAI-compatible APIs to turn OCR-mangled books into readable, sane text. Because nobody wants to read the *literal* results of a neural network.*  

```
        o    .   _     .
          .     (_)         o
   o      ____            _       o
  _   ,-/   /)))  .   o  (_)   .
 (_)  \_\  ( e(     O             _
 o       \/' _/   ,_ ,  o   o    (_)
  . O    _/ (_   / _/      .  ,        o
     o8o/    \\_/ / ,-.  ,oO8/( -TT
    o8o8O | } }  / /   \Oo8OOo8Oo||     O
   Oo(""o8"""""""""""""""8oo""""""")
  _   `\`'                  `'   /'   o
 (_)    \                       /    _   .
      O  \           _         /    (_)
o   .     `-. .----<(o)_--. .-'
   --------(_/------(_<_/--\_)--------hjw
```

```bash
λ awful_text_sanitizer --help
Clean up excerpts from books formatted as txt

Usage: awful_text_sanitizer [OPTIONS] --input <INPUT_DIR> --output <OUTPUT_DIR>

Options:
  -i, --input <INPUT_DIR>    Path to directory of txt files
  -o, --output <OUTPUT_DIR>  Path to directory where yaml files will be written
      --config <CONFIG>...   Configuration files (can specify multiple)
  -h, --help                 Print help
```

---

## 📚 What Is This?  

This is **`awful_book_sanitizer`**, a command-line tool designed to clean up text excerpts from books that were *too spooky* for OCR.  

**Key features:**  
- **Asynchronous processing** with multiple configurations (for different LLMs/APIs).  
- **Chunked text splitting** to avoid overwhelming models.  
- **YAML output format**, so you can later analyze sanity or just read the text.  
- **Exponential backoff** to handle API failures like a seasoned ghostbuster.  

*Despite its ominous name, it's actually pretty awesome.*  

---

## 🧩 How It Works

1. **Input**: A directory of `.txt` files (probably from OCR).  
2. **Chunk It Up**: Split text into 500-token chunks (a number chosen because it felt right).  
3. **Send to LLM**: Use a conversational template (like "You are a librarian who fixes typos") to ask the model to *sanitize* the text.  
4. **Output**: YAML files with chunks of clean text (or nope, if the API threw a tantrum).  

*Think of it as a magical wand that turns "This is a really bad word" into "That's actually the correct spelling."*

---

## 🧪 Example Usage  

```bash
awful_book_sanitizer --i /path/to/ocr-books --o /path/to/output --c llama-cpp-config.yaml google-colab-config.yaml
```

**What this does**:  
- `-i` = input directory (e.g., "ocr_books").  
- `-o` = output directory (will create YAML files like `book1.yaml`).  
- `-c` = configuration files for different LLMs.  

**Bonus tip**: If your books are "too spooky," try adding `$RUST_LOG=info` to see the program's internal monologue.  

---

## 🕌 Architecture (In A Nutshell)  

- **Text Splitter**: Divides text into chunks of 500 tokens (like a greedy librarian).  
- **Template Engine**: Sends prompts to the LLM (e.g., "You are a librarian who fixes typos").  
- **Async Threads**: Processes multiple configurations simultaneously (like having 20 assistants at once).  

*This is not just a sanitizer—it's a high-stakes collaboration between humans and AI.*  

---

## 🧑‍💻 Contributing & Feedback  

- **Report bugs**: The program's "sanity" isn't guaranteed. If your output is *too* sane, something went wrong.  
- **Suggest improvements**: We're a team of "empathetic sociopaths" trying to make the best of things.  
- **Share your data**: If you have OCR-mangled books, submit them (but only if they’re not spooky).  

*Just remember: The goal isn’t to fix every typo. It’s to make sure your books are at least *legible*.  

---

## 📌 Credits  

- **Core Authors**: Thomas Gentry \<thomas@awulsec.com\> (the human cranking the flywheel that makes this stuff)
- **Rust Engine**: Written in Rust with `tokio`, `serde`, and `clap`.  
- **LLM Templates**: Based on OpenAI-compatible endpoints (e.g., `llama-cpp @ localhost` or `Google Colab`).  
- **Inspiration**: All the bad OCR results you've ever encountered.  

---

## 😸 Fun Fact  

The hardest part about finetuning an LLM is managing all of the input and output files. 🤔

*You’re welcome to email me with creative solutions.*  

--- 

## 🧪 Want to Try It?  

1. **Install dependencies**:  
   ```bash
   cargo install awful_book_sanitizer
   ```

2. **Run it**:  
   ```bash
   awful_book_sanitizer -i books -o output -c configs.yaml
   ```

3. **Check your YAML files**:  
   ```bash
   cat results/book1.yaml
   ```

*Please try to refrain from creating accidental heresy.*  

--- 

## 🧠 Final Thoughts  

This program is a love letter to bad OCR and the power of LLMs. It’s not perfect, but it’s a step toward *sanity* for your books.  

**Remember**: The goal isn’t to make the text perfect—it’s to make it *useful enough*.  

*Now go forth and sanitize!* 🧪📚
