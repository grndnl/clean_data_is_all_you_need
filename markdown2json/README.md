# Markdown to JSON converter

## JSON Schema

The following is the json schema that describes the data and metadata extracted from a scientific paper using our pipeline: 
```json
{
    "paper_id": "string",
    "title": "string",
    "paper_text": [
        {
            "section_name": "string",
            "section_text": "string",
            "section_page": "int",
            "section_location": [...]
      },
        ...
    ],
    "figures": [
        {
            "figure_id": "string",
            "figure_page": "int",
            "figure_location": [...]
        },
        ...
    ],
    "tables": [
        {
            "table_id": "string",
            "table_text": "string",
            "table_page": "int",
            "table_location": [...]
        },
        ...
    ],
    "equations": [
        {
            "equation_id": "string",
            "equation_latex": "string",
            "equation_page": "int",
            "equation_location": [...]
        },
        ...
    ],
}
```