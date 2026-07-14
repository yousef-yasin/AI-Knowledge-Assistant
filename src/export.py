#export file means that we will export files in different formats like txt and markdown


from datetime import datetime #to generate timestamped filenames
#mean that each file we create will have a unique name based on the current date and time, preventing overwriting of previous exports
#have name like the date and time of creation

from pathlib import Path #to handle file paths and directories
#mean that we will make variables that represent file paths and directories

from typing import Any #to allow for flexible typing of sources and conversation data
#mean that wee can use any different data types in our code


EXPORTS_DIR = Path("exports") #to make a variable that represents the directory where exported files will be saved

 
#CLASS 1
def ensure_exports_directory() -> None:  #CHECK IF THE EXPORTS DIRECTORY EXISTS, IF NOT CREATE IT
    #None mean that this function does not return any value
    """
    Create the exports directory if it does not already exist.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True) #to create the exports directory if it doesn't exist
 #parents=True mean that if the parent directories do not exist, they will be created as well
 #exist_ok=True mean that if the directory already exists, no error will be raised


#CLASS 2
def generate_filename(extension: str) -> Path: #to generate a unique filename for the exported file based on the current date and time
    #path mean that this function will return a Path object representing the generated filename
    """
    Generate a unique export filename using the current date and time.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #to get the current date and time and format it as a string
    return EXPORTS_DIR / f"conversation_{timestamp}.{extension}"
 #to make the filename like "conversation_2023-10-05_14-30-00.txt" or "conversation_2023-10-05_14-30-00.md"
    
#CLASS 3
def format_sources(sources: list[Any]) -> list[str]: #to format the sources into readable text lines
    #mean that this function will return a list of strings representing the formatted sources
    """
    Convert sources into readable text lines.

    Sources may be strings or dictionaries.
    """
    formatted_sources: list[str] = [] #to make a list to store the formatted sources

    for source in sources: #to iterate through each source in the sources list
        if isinstance(source, str): #to check if the source is a string
            formatted_sources.append(source) #to add the string source to the formatted_sources list
#mean that if the source is a string, we will just add it to the formatted_sources list as is

        elif isinstance(source, dict): #if the source dictionary, we will extract the relevant information and  format it 

            document_name = source.get("document_name", "Unknown document") #to get the document name from the source dictionary
            page_number = source.get("page_number") #to get the page number from the source dictionary
            chunk_number = source.get("chunk_number") #to get the chunk number from the source dictionary

            source_text = document_name #to initialize the source text with the document name

            if page_number is not None:
                source_text += f" - Page {page_number}" #to append the page number to the source text if it exists

            if chunk_number is not None:
                source_text += f" - Chunk {chunk_number}" #to append the chunk number to the source text if it exists

            formatted_sources.append(source_text)

        else:
            formatted_sources.append(str(source)) #to convert it to a string and add it to the formatted_sources list

    return formatted_sources



#CLASS 4
def export_to_txt(conversation: list[dict[str, Any]]) -> str: #to export the conversation to a TXT file
    """
    Export a conversation to a TXT file.

    Expected conversation format:

    [
        {
            "question": "What is the annual leave policy?",
            "answer": "Employees receive 14 annual leave days.",
            "sources": [
                {
                    "document_name": "Employee Handbook.pdf",
                    "page_number": 15,
                    "chunk_number": 4
                }
            ],
            "confidence": 94
        }
    ]
    """
    ensure_exports_directory() #to ensure that the exports directory exists before attempting to write the fileS
    file_path = generate_filename("txt")

    with file_path.open("w", encoding="utf-8") as file:
        #open the file in write mode with UTF-8 encoding to handle any special characters in the conversation text
        file.write("AI Knowledge Assistant Conversation\n")
        file.write("=" * 45 + "\n\n")
#to make a header for the exported conversation file, followed by a line of equal signs for visual separation
       
        for index, message in enumerate(conversation, start=1):
            question = message.get("question", "Question unavailable")
            answer = message.get("answer", "Answer unavailable")
            sources = format_sources(message.get("sources", []))
            confidence = message.get("confidence", "N/A")

            file.write(f"Question {index}\n") #print the question number
            file.write("-" * 20 + "\n")
            file.write(f"{question}\n\n") #WRITE THE QUESTION TEXT

            file.write("Answer\n")
            file.write("-" * 20 + "\n")
            file.write(f"{answer}\n\n") #write the answer text

            file.write("Sources\n") #PRINT THE SOURCES HEADER
            file.write("-" * 20 + "\n")

            if sources:
                for source in sources:
                    file.write(f"- {source}\n")
            else:
                file.write("- No sources available\n")

            file.write(f"\nConfidence: {confidence}%\n")
            file.write("\n" + "=" * 45 + "\n\n")

    return str(file_path) #retuern the path to the exported TXT file as a string



#CLASS 5
def export_to_markdown(conversation: list[dict[str, Any]]) -> str:
    #MAKE A FUNCTION THAT EXPORTS THE CONVERSATION TO A MARKDOWN FILE
    """
    Export a conversation to a Markdown file.
    """
    ensure_exports_directory()
    file_path = generate_filename("md")

    with file_path.open("w", encoding="utf-8") as file:
        file.write("# AI Knowledge Assistant Conversation\n\n")

        for index, message in enumerate(conversation, start=1):
            question = message.get("question", "Question unavailable")
            answer = message.get("answer", "Answer unavailable")
            sources = format_sources(message.get("sources", []))
            confidence = message.get("confidence", "N/A")

            file.write(f"## Question {index}\n\n")
            file.write(f"{question}\n\n")

            file.write("### Answer\n\n")
            file.write(f"{answer}\n\n")

            file.write("### Sources\n\n")

            if sources:
                for source in sources:
                    file.write(f"- {source}\n")
            else:
                file.write("- No sources available\n")

            file.write(f"\n**Confidence:** {confidence}%\n\n")
            file.write("---\n\n")

    return str(file_path)


#class 6
def export_conversation(
    conversation: list[dict[str, Any]], #allow for flexible typing of sources and conversation data
    export_format: str #type of export format, either 'txt' or 'md' (markdown)
) -> str:
    """
    Export the conversation using the selected format.

    Supported formats:
    - txt
    - md
    - markdown
    """
    normalized_format = export_format.strip().lower() #DELETE ANY LEADING OR TRAILING WHITESPACE AND CONVERT THE FORMAT TO LOWERCASE FOR CONSISTENCY

    if normalized_format == "txt":
        return export_to_txt(conversation)

    if normalized_format in {"md", "markdown"}:
        return export_to_markdown(conversation)

    raise ValueError(
        "Unsupported export format. Use 'txt', 'md', or 'markdown'."
    )


if __name__ == "__main__":
    sample_conversation = [ #make a sample conversation to test the export functions
        {
            "question": "What is the annual leave policy?",
            "answer": "Employees receive 14 annual leave days.",
            "sources": [
                {
                    "document_name": "Employee Handbook.pdf",
                    "page_number": 15,
                    "chunk_number": 4
                },
                {
                    "document_name": "HR Policy.pdf",
                    "page_number": 8,
                    "chunk_number": 2
                }
            ],
            "confidence": 94
        },
        {
            "question": "What are the working hours?",
            "answer": "The official working hours are from 9:00 AM to 5:00 PM.",
            "sources": [
                {
                    "document_name": "Company Policy.txt",
                    "chunk_number": 1
                }
            ],
            "confidence": 91
        }
    ]
 
    txt_path = export_to_txt(sample_conversation) #save the sample conversation to a TXT file
    markdown_path = export_to_markdown(sample_conversation) #make a function that saves the sample conversation to a Markdown file

    print(f"TXT file created successfully: {txt_path}")
    print(f"Markdown file created successfully: {markdown_path}")