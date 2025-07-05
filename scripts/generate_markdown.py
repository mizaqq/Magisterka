#!/usr/bin/env python3
"""
Markdown Generator Script

Takes answer.json containing question and answer data and generates
properly formatted markdown file with appropriate filename.
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict


def load_answer_data() -> Dict:
    """Load and validate JSON from fixed path answer.json"""
    try:
        with open('answer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['question_number', 'question_text', 'answer_text', 'status', 'word_count', 'generated_at']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate data types and ranges
        if not isinstance(data['question_number'], int) or not (1 <= data['question_number'] <= 50):
            raise ValueError("Question number must be integer between 1-50")

        if data['status'] not in ['completed', 'draft', 'pending']:
            raise ValueError("Status must be one of: completed, draft, pending")

        return data

    except FileNotFoundError:
        raise FileNotFoundError("answer.json not found in current directory")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in answer.json")


def create_slug(text: str, max_length: int = 30) -> str:
    """Create URL-friendly slug from question text"""
    # Remove special characters and convert to lowercase
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace spaces and hyphens with underscores
    slug = re.sub(r'[-\s]+', '_', slug)
    # Remove multiple underscores
    slug = re.sub(r'_+', '_', slug)
    # Remove leading/trailing underscores
    slug = slug.strip('_')

    # Limit length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('_')

    return slug


def format_markdown_content(answer_data: Dict) -> str:
    """Format the markdown content with question and answer"""
    question_number = answer_data['question_number']
    question_text = answer_data['question_text']
    answer_text = answer_data['answer_text']
    status = answer_data['status']
    word_count = answer_data['word_count']
    generated_at = answer_data['generated_at']

    # Status emoji
    status_emoji = {'completed': '✅', 'draft': '📝', 'pending': '⏳'}

    content = f"""# Question {question_number}: {question_text}

**Status:** {status_emoji.get(status, '❓')} {status.title()} | **Words:** {word_count} | **Generated:** {generated_at}

---

## Answer

{answer_text}

---
"""

    return content


def save_markdown(content: str, question_number: int, question_text: str) -> str:
    """Save formatted markdown with auto-naming"""
    # Create answers directory if it doesn't exist
    answers_dir = Path('answers')
    answers_dir.mkdir(exist_ok=True)

    # Generate filename slug
    slug = create_slug(question_text)
    filename = f"{question_number:02d}_{slug}.md"

    output_path = answers_dir / filename

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Markdown generated successfully: {output_path}")
        return str(output_path)

    except Exception as e:
        raise IOError(f"Failed to save markdown: {e}")


def main():
    """Main execution function"""
    try:
        print("🔍 Loading answer data...")
        answer_data = load_answer_data()

        print(f"📝 Processing Question {answer_data['question_number']}: {answer_data['question_text'][:60]}...")

        print("📄 Formatting markdown content...")
        content = format_markdown_content(answer_data)

        print("💾 Saving markdown...")
        output_path = save_markdown(content, answer_data['question_number'], answer_data['question_text'])

        print(f"\n🎉 Success! Generated markdown file")
        print(f"📁 Output file: {output_path}")
        print(f"📊 Content: {answer_data['word_count']} words, Status: {answer_data['status']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
