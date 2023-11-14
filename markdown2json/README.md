# Markdown to JSON converter

## JSON Schema

The following is the json schema that describes the data and metadata extracted from a scientific paper using our pipeline: 
```json
{
    "paper_id": "string",
    "title": "string",
    "paper_text": [
        0 : {
            "section_name": "string",
            "section_text": "string",
            "section_page": "int",
            "section_location": [...]
      },
        ...
    ]
}
```