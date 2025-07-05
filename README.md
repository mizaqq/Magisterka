# Thesis Answer Generator

A Python tool for generating well-formatted documents containing thesis answers that follow academic standards and the established cursor-based generation rules.

## Features

- ✅ **Rule Validation**: Automatically validates answers against the 10-sentence limit and other academic standards
- 📊 **Statistics**: Provides detailed statistics about your answer collection
- 📄 **Markdown Generation**: Creates clean, readable markdown documents
- 📋 **Interactive Interface**: Easy-to-use command-line interface for managing questions
- 🎯 **Academic Focus**: Tailored for Big Data Analytics Master's students
- 🔧 **Tool Tracking**: Tracks and displays mentioned tools and methods
- 📝 **Reusable**: Single script handles all questions without creating multiple files

## Installation

1. No core dependencies needed for markdown generation
2. Optional: Install reportlab for PDF generation: `pip install reportlab`
3. The tool is ready to use!

## Quick Start

### Interactive Mode
```bash
python thesis_answer_manager.py
```

### Programmatic Mode
```python
from thesis_answer_manager import ThesisAnswerManager

# Create manager
manager = ThesisAnswerManager()

# Add answer programmatically
manager.add_answer_programmatic(
    question_number=1,
    answer="Your answer here (2-10 sentences following the rules)",
    disciplinary_context="Economics, Mathematics",
    tools_mentioned=["tool1", "tool2"]
)

# Validate and generate markdown
manager.validate_all_answers()
filename = manager.generate_markdown("my_answers.md")
manager.mark_questions_done([1])
```

## Rules Enforced

The tool automatically enforces the following rules from `general_rules.mdc`:

1. **10-sentence maximum** per answer
2. **Academic tone** and professional formatting
3. **Disciplinary context** tracking (economics, mathematics, programming, statistics)
4. **Tool/method references** documentation
5. **Standalone answers** that don't require context from other questions
6. **Structured format**: intro → core points → conclusion

## Usage Examples

### Interactive Commands
```bash
# Run the interactive manager
python thesis_answer_manager.py

# Available commands:
# 1. Show all questions
# 2. Show pending questions  
# 3. Answer question interactively
# 4. Add pre-written answer
# 5. Validate all answers
# 6. Show statistics
# 7. Generate markdown
# 8. Mark questions as done
# 9. Exit
```

### Programmatic Usage
```python
from thesis_answer_manager import ThesisAnswerManager

manager = ThesisAnswerManager()

# Show questions
manager.show_questions()
manager.show_questions("PENDING")  # Only pending questions

# Add answers
manager.add_answer_programmatic(
    question_number=5,
    answer="Your comprehensive answer here...",
    disciplinary_context="Statistics, Machine Learning",
    tools_mentioned=["regression", "cross-validation"]
)

# Validate and generate
manager.validate_all_answers()
manager.show_statistics()
manager.generate_markdown("my_thesis_answers.md")
```

### Example Run
```bash
# Run with example data
python thesis_answer_manager.py example
```

## File Structure

- `thesis_answer_generator.py` - Core answer validation and generation
- `thesis_answer_manager.py` - Main reusable interface
- `questions/questions.md` - Source questions file
- `requirements.txt` - Optional dependencies
- `README.md` - This documentation

## Academic Guidelines

### Answer Structure
1. **Introduction**: Brief sentence rephrasing the question
2. **Core Points**: 2-4 main explanatory points
3. **Conclusion**: Summary or concluding insight (optional)

### Best Practices
- Keep answers between 2-10 sentences
- Reference specific tools and methods
- Ground answers in relevant academic disciplines
- Use concrete examples over abstract generalizations
- Maintain professional academic tone

## Example Output

The tool generates PDFs with:
- Clean academic formatting
- Professional typography
- Proper spacing and structure
- Question numbering
- Disciplinary context sections
- Tool/method references
- Summary statistics

## Contributing

Feel free to extend the tool with additional features like:
- Custom PDF styling
- Export to other formats (Word, LaTeX)
- Integration with reference management
- Advanced validation rules
