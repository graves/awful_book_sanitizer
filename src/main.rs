/// `awful_book_sanitizer`
///
/// This command-line interface is used to clean up book excerpts from `.txt` files.
/// Books processed by OCR are typically full of bad characters, mispelled words,
/// and poor grammar.
///
/// This program processes text files, splits them into 500 token chunks, and uses a
/// conversational template to ask a Large Language Model running at an OpenAI
/// compatible endpoint to sanitize (make sane) the text.
///
/// The result of each query is appended to an array in a YAML file named
/// after the input file.
///
/// You can pass in multiple configuration files which contain the API url you wish
/// to send the sanitization request to. The requests will be processed asynchronously
/// using a thread for each configuration file. This enables you to leverage multiple
/// LLM instances simultaneosly.
///
/// # Example Usage
/// ```bash
/// awful_book_sanitizer --i /path/to/input --o /path/to/output --c llama-cpp-config.yaml google-colab-config.yaml
/// ```
use std::path::PathBuf;
use std::{fs, time::Duration};

use awful_aj::{
    api::ask,
    config::AwfulJadeConfig,
    template::{self, ChatTemplate},
};
use clap::{Parser, command};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::Write as IoWrite;
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;
use tokio::task::spawn_blocking;
use tokio::time::sleep;

/// CLI arguments
#[derive(Parser, Debug)]
#[command(name = "awful_book_sanitizer")]
/// Clean up excerpts from books formatted as txt
#[command(about = "Clean up excerpts from books formatted as txt", long_about = None)]
struct Args {
    /// Path to directory of txt files
    #[arg(short, long = "i")]
    input_dir: PathBuf,

    /// Path to directory where yaml files will be written
    #[arg(short, long = "o")]
    output_dir: PathBuf,

    /// Configuration files (can specify multiple)
    #[arg(long = "c", num_args = 1..)]
    config: Vec<PathBuf>,
}

/// Data structure for sanitized book excerpts
#[derive(Debug, Deserialize, Serialize)]
pub struct BookChunk {
    /// Sanitized text from a book excerpt
    pub sanitizedBookExcerpt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Parse command-line arguments
    let args = Args::parse();

    // Clone paths to avoid moving them during spawning
    let input_dir_path = args.input_dir.clone();
    let output_dir_path = args.output_dir.to_str().unwrap().to_string();

    // Spawn tasks for each configuration file
    let mut handles = Vec::new();

    for config_path in &args.config {
        // Load configuration from file
        let config = awful_aj::config::load_config(config_path.to_str().unwrap())
            .map_err(|e| format!("Config load error: {e}"))?;

        // Clone paths for safe use in spawned tasks
        let input_clone = input_dir_path.clone();
        let output_clone = output_dir_path.clone();

        // Spawn a blocking task to process files
        handles.push(spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                process_files(&input_clone, &output_clone, config)
                    .await
                    .map_err(|e| -> Box<dyn Error + Send + Sync> { e.into() })
            })
        }));
    }

    // Wait for all tasks to complete
    for handle in handles {
        if let Err(e) = handle.await? {
            eprintln!("Error in task: {}", e);
        }
    }

    Ok(())
}

/// Process text files and sanitize their content
///
/// # Parameters
/// - `input_dir`: Path to directory containing `.txt` files to process
///
/// - `output_dir_path`: Base path where YAML output files will be written (e.g., `/path/to/output`)
///
/// - `config`: Configuration settings for the sanitization process
///
/// # Return Value
/// - `Result<(), String>`: Returns `Ok(())` on success, or an error message
///
/// # Example Usage
/// ```rust
/// let result = process_files("/path/to/input", "/path/to/output", config).await;
/// if let Err(err) = result {
///     eprintln!("Error: {}", err);
/// }
/// ```
async fn process_files(
    input_dir: &PathBuf,
    output_dir_path: &str,
    config: AwfulJadeConfig,
) -> Result<(), String> {
    // Initialize tokenizer for tokenization
    let tokenizer = cl100k_base().map_err(|e| e.to_string())?;
    let max_tokens = 500;

    // Configure text splitter to chunk content
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    // Load template for sanitization
    let template = template::load_template("book_txt_sanitizer")
        .await
        .map_err(|e| format!("Template load error: {e}"))?;

    // Process each file in the input directory
    for entry in fs::read_dir(input_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = &entry.path();

        // Check if the file is a `.txt` text file
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let filename = path.file_name().unwrap().to_string_lossy();
            let mut yaml_path = format!("{}/{}.yaml", output_dir_path, filename);

            // Open YAML file for writing
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&yaml_path)
                .map_err(|e| e.to_string())?;

            // Write YAML header
            writeln!(file, "chunks:").map_err(|e| e.to_string())?; // Write YAML header

            // Read and process the text content
            let contents = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let chunks: Vec<_> = splitter.chunks(&contents).collect();

            // Process each chunk
            for chunk in chunks {
                let book_chunk = fetch_with_backoff(&config, &chunk, &template)
                    .await
                    .map_err(|e| e.to_string())?;

                if let Some(sanitized_text) = book_chunk {
                    // Write sanitized content to YAML
                    write_row_to_file(sanitized_text, &mut yaml_path).map_err(|e| e.to_string())?;
                }
            }
        }
    }

    Ok(())
}

/// Write a sanitized text chunk to YAML file
///
/// # Parameters
/// - `chunk`: The sanitized text content to write
///
/// - `yaml_path`: Path to the YAML file (modified by reference)
///
/// # Return Value
/// - `Result<(), Box<dyn Error + Send + Sync>>`:
///   - `Ok(())` on success
///
/// # Example Usage
/// ```rust
/// let mut yaml_path = "/path/to/output.yaml".to_string();
/// write_row_to_file("Sanitized content", &mut yaml_path).unwrap();
/// ```
pub fn write_row_to_file(
    chunk: String,
    yaml_path: &mut String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&yaml_path)?; // Open YAML file for appending

    // Write YAML line with content
    writeln!(file, "\t- |-").map_err(|e| e.to_string())?;
    for line in chunk.lines() {
        writeln!(file, "\t\t{}", line).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Fetch sanitized text with exponential backoff for reliability
///
/// # Parameters
/// - `config`: Configuration settings for the sanitization process
///
/// - `chunk`: The text chunk to sanitize (as a string)
///
/// - `template`: Template for sanitization instructions
///
/// # Return Value
/// - `Result<Option<String>, Box<dyn Error + Send + Sync>>`:
///   - `Ok(Some(sanitized_text))` if successful
///   - `Ok(None)` if no response or empty result
///   - `Err(errorMsg)` for errors (e.g., API failures, configuration issues)
///
/// # Example Usage
/// ```rust
/// let config = ...; // Loaded configuration
/// let chunk = "Sample text to sanitize";
/// let template = ChatTemplate::new("Ask");
///
/// match fetch_with_backoff(&config, chunk, &template).await {
///     Ok(Some(text)) => println!("Sanitized: {}", text),
///     Ok(None) => println!("No response from API"),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
///
const MAX_RETRIES: u32 = 5;
const BASE_DELAY_MS: u64 = 500;

async fn fetch_with_backoff(
    config: &AwfulJadeConfig,
    chunk: &str,
    template: &ChatTemplate,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    for attempt in 0..=MAX_RETRIES {
        let res = ask(config, chunk.to_string(), template, None, None).await;

        match res {
            Ok(res) => {
                if res != "{}" {
                    let book_chunk: BookChunk = serde_json::from_str(&res)?;
                    return Ok(Some(book_chunk.sanitizedBookExcerpt));
                } else {
                    return Ok(None);
                }
            }
            Err(err) => {
                eprintln!("Request failed: {}", err); // Log error
            }
        }

        if attempt < MAX_RETRIES {
            let backoff = BASE_DELAY_MS * (2u64.pow(attempt));
            eprintln!("Retrying in {}ms...", backoff);
            sleep(Duration::from_millis(backoff)).await;
        }
    }

    Err("All retries failed".into())
}
