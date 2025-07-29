#!/usr/bin/env python3
"""Create a test PDF file for RAG testing"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from pathlib import Path

def create_test_pdf():
    output_path = Path('test_documents/rag_test_document.pdf')
    
    # Create document
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
    )
    
    story = []
    
    # Add title
    title = Paragraph("RAG System PDF Test Document", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add introduction
    intro_text = """
    This is a test PDF document created to demonstrate the RAG (Retrieval-Augmented Generation) 
    system's ability to process and extract text from PDF files. The document contains multiple 
    pages with various types of content that should be indexed and made searchable through the 
    RAG system.
    """
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add section header
    story.append(Paragraph("RAG System Architecture", styles['Heading2']))
    
    # Add technical content
    arch_text = """
    The RAG system employs a sophisticated architecture that combines document processing, 
    text embedding, and semantic search capabilities. The system can handle multiple document 
    formats including PDF, DOCX, and various text-based files.
    
    Key components include:
    • Document processors for different file types
    • Sentence transformer models for creating embeddings
    • SQLite database for knowledge storage
    • Semantic search with cosine similarity
    • Automatic text chunking for large documents
    """
    story.append(Paragraph(arch_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add a table
    story.append(Paragraph("System Components", styles['Heading2']))
    
    table_data = [
        ['Component', 'Technology', 'Purpose'],
        ['Text Extraction', 'PyPDF2, python-docx', 'Extract text from documents'],
        ['Embeddings', 'sentence-transformers', 'Create semantic vectors'],
        ['Storage', 'SQLite', 'Store documents and embeddings'],
        ['Search', 'Cosine Similarity', 'Find relevant content'],
        ['Chunking', 'Smart Text Splitting', 'Handle large documents']
    ]
    
    table = Table(table_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Add usage section
    story.append(Paragraph("Usage Instructions", styles['Heading2']))
    
    usage_text = """
    To use the RAG system with document support:
    
    1. Install required dependencies: pip install sentence-transformers numpy python-docx PyPDF2
    2. Enable RAG for an assistant: coder enable rag myassistant
    3. Index documents: coder index myassistant /path/to/documents
    4. Query the system: Ask questions about your indexed content
    
    The system will automatically extract text from PDF and DOCX files, create embeddings, 
    and make the content searchable through natural language queries.
    """
    story.append(Paragraph(usage_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add second page content
    story.append(Paragraph("Advanced Features", styles['Heading2']))
    
    advanced_text = """
    The RAG system includes several advanced features for document processing:
    
    Document Chunking: Large documents are automatically split into smaller, overlapping 
    chunks to improve retrieval accuracy. The system uses smart sentence boundary detection 
    to ensure chunks break at natural points.
    
    Metadata Extraction: The system extracts and stores metadata from documents, including 
    author information, creation dates, page counts, and document structure information.
    
    Fallback Search: When embedding models are unavailable, the system falls back to 
    keyword-based search to ensure functionality is maintained.
    
    Source Attribution: Retrieved content includes source information, allowing users to 
    trace answers back to specific documents and even page or section numbers.
    """
    story.append(Paragraph(advanced_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add conclusion
    story.append(Paragraph("Conclusion", styles['Heading2']))
    
    conclusion_text = """
    This PDF document serves as a comprehensive test for the RAG system's PDF processing 
    capabilities. It contains structured content, tables, multiple sections, and various 
    types of information that should be extractable and searchable through the RAG system.
    
    When properly indexed, users should be able to ask questions about RAG architecture, 
    system components, usage instructions, or advanced features and receive relevant 
    answers with source attribution to this document.
    """
    story.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"Created test PDF file: {output_path}")

if __name__ == "__main__":
    create_test_pdf()