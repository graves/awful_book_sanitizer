use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::str::{Utf8Chunk, Utf8Chunks};
use std::{fs, time::Duration};

use awful_aj::{
    api::ask,
    config::{self, AwfulJadeConfig},
    config_dir,
    models::AwfulConfig,
    template::{self, ChatTemplate},
};
use clap::Parser;
use clap::command;
use serde::{Deserialize, Serialize};
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;
use tokio::{io::AsyncWriteExt, time::sleep};

/// CLI arguments
#[derive(Parser, Debug)]
#[command(name = "awful_book_sanitizer")]
#[command(about = "Clean up excerpts from books formatted as txt", long_about = None)]
struct Args {
    /// Path to directory of txt files
    #[arg(short, long)]
    input_dir: PathBuf,
    /// Path to directory where yaml files will be written
    #[arg(short, long)]
    output_dir: PathBuf,
    /// Configuration file
    #[arg(short, long)]
    config: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct BookChunk {
    pub sanitizedBookExcerpt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input_dir_path = args.input_dir;
    let conf_file = args.config;
    let output_dir_path = args.output_dir.display();

    let tokenizer = cl100k_base().unwrap();
    let max_tokens = 500; // Context size is 32k
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    let template = template::load_template("book_txt_sanitizer").await?;
    let config =
        config::load_config(conf_file.to_str().expect("Not a valid config filename")).unwrap();

    for entry in fs::read_dir(input_dir_path)? {
        let entry = entry?;
        let path = entry.path();

        // Only process .txt files
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let filename = path.file_name().unwrap().to_string_lossy();
            let contents = fs::read_to_string(&path)?;
            println!("File: {}\n", filename);
            let mut yaml: String = "chunks:\n".to_string();
            let mut yaml_path = format!("{output_dir_path}/{filename}.yaml");
            let chunks = splitter.chunks(&contents);
            let total_count = chunks.count();
            let chunks = splitter.chunks(&contents);

            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&yaml_path)?;
            let _res = writeln!(file, "chunks:");

            for (count, chunk) in chunks.into_iter().enumerate() {
                println!("Processing {}/{} chunks.", count + 1, total_count);
                let book_chunk = fetch_with_backoff(&config, chunk, &template).await?;

                if let Some(santitized_text) = book_chunk {
                    println!("SANITIZED: {santitized_text}");
                    write_row_to_file(santitized_text, &mut yaml_path);
                }
            }
        }
    }

    Ok(())
}

pub fn write_row_to_file(
    chunk: String,
    yaml_path: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&yaml_path)?;

    let _res = writeln!(file, "\t- |-");

    for line in chunk.lines() {
        let _res = writeln!(file, "\t\t{}", line);
    }

    Ok(())
}

const MAX_RETRIES: u32 = 5;
const BASE_DELAY_MS: u64 = 500;

async fn fetch_with_backoff(
    config: &AwfulJadeConfig,
    chunk: &str,
    template: &ChatTemplate,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    for attempt in 0..=MAX_RETRIES {
        let res = ask(&config, chunk.to_string(), &template, None, None).await;

        match res {
            Ok(res) => {
                println!("res: {:?}", res);
                if res != "{}" {
                    let book_chunk: Result<BookChunk, serde_json::Error> =
                        serde_json::from_str(&res);
                    return Ok(Some(book_chunk.unwrap().sanitizedBookExcerpt));
                } else {
                    return Ok(None);
                }
            }
            Err(err) => {
                eprintln!("Request failed: {}", err);
            }
        }

        if attempt < MAX_RETRIES {
            let backoff = BASE_DELAY_MS * (2u64.pow(attempt));
            eprintln!("Retrying in {}ms...", backoff);
            sleep(Duration::from_millis(backoff)).await;
        }
    }

    Err("Hyper timeout".into())
}
