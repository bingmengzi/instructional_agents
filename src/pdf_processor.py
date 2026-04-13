"""
PDF Slide Deck Processor
Process PDF slide deck files with support for extracting content by chapter/requirement
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available")


class PDFSlideProcessor:
    """Process PDF slide files with on-demand extraction"""
    
    def __init__(self, output_dir: str = "knowledge_base"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def store_pdf_files(self, files: List[Path], storage_id: str) -> Dict[str, Any]:
        """
        Store PDF files (without processing immediately)

        Args:
            files: List of PDF files
            storage_id: Storage identifier

        Returns:
            Storage information
        """
        storage_dir = self.output_dir / "temp_storage" / storage_id
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        stored_files = []
        for pdf_file in files:
            if pdf_file.suffix.lower() == '.pdf':
                dest_path = storage_dir / pdf_file.name
                # Copy file to storage directory
                shutil.copy2(pdf_file, dest_path)
                stored_files.append({
                    "filename": pdf_file.name,
                    "stored_path": str(dest_path),
                    "size": os.path.getsize(dest_path)
                })
        
        metadata = {
            "storage_id": storage_id,
            "stored_at": datetime.now().isoformat(),
            "total_files": len(stored_files),
            "files": stored_files
        }
        
        # Save metadata
        metadata_file = storage_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def extract_by_requirement(
        self, 
        storage_id: str,
        user_requirements: str,
        target_chapters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract relevant PDF content based on user requirements

        Args:
            storage_id: Storage identifier
            user_requirements: Description of user requirements
            target_chapters: List of target chapters (e.g. ["Chapter 1", "Chapter 3"]); if None, analyze all

        Returns:
            Extracted content data
        """
        storage_dir = self.output_dir / "temp_storage" / storage_id
        
        if not storage_dir.exists():
            raise ValueError(f"Storage {storage_id} not found")
        
        # Load metadata
        metadata_file = storage_dir / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Get all PDF files
        pdf_files = [Path(f["stored_path"]) for f in metadata["files"]]
        
        # If chapters are specified, identify relevant files first
        if target_chapters:
            relevant_files = self._identify_relevant_files(pdf_files, target_chapters)
        else:
            # If user requests optimization of the entire course, analyze all chapters
            relevant_files = pdf_files
        
        # Extract relevant content
        extracted_slides = []
        for pdf_file in relevant_files:
            try:
                slide_data = self.process_single_pdf(
                    str(pdf_file),
                    user_requirements=user_requirements,
                    target_chapters=target_chapters
                )
                extracted_slides.append(slide_data)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                extracted_slides.append({
                    "filename": pdf_file.name,
                    "error": str(e)
                })
        
        return {
            "storage_id": storage_id,
            "extracted_at": datetime.now().isoformat(),
            "user_requirements": user_requirements,
            "target_chapters": target_chapters,
            "total_extracted_files": len([s for s in extracted_slides if "error" not in s]),
            "slides": extracted_slides
        }
    
    def _identify_relevant_files(
        self, 
        pdf_files: List[Path], 
        target_chapters: List[str]
    ) -> List[Path]:
        """
        Identify PDF files containing target chapters

        Args:
            pdf_files: List of PDF files
            target_chapters: List of target chapters

        Returns:
            List of relevant PDF files
        """
        relevant_files = []
        
        for pdf_file in pdf_files:
            # Quick scan of PDF title and first page to check for chapter information
            try:
                # Check filename
                filename_match = any(
                    self._match_chapter_in_text(pdf_file.stem, chapter) 
                    for chapter in target_chapters
                )
                
                if filename_match:
                    relevant_files.append(pdf_file)
                    continue
                
                # Check titles in the first few pages
                if PDFPLUMBER_AVAILABLE:
                    with pdfplumber.open(pdf_file) as pdf:
                        for page_num in range(min(3, len(pdf.pages))):
                            page = pdf.pages[page_num]
                            text = page.extract_text() or ""
                            
                            # Check if it contains chapter keywords
                            if any(
                                self._match_chapter_in_text(text, chapter) 
                                for chapter in target_chapters
                            ):
                                relevant_files.append(pdf_file)
                                break
            except Exception as e:
                print(f"Warning: Could not scan {pdf_file.name}: {e}")
                # If scanning fails, include the file by default
                relevant_files.append(pdf_file)
        
        return relevant_files
    
    def _match_chapter_in_text(self, text: str, chapter_pattern: str) -> bool:
        """Check if text matches a chapter pattern"""
        # Supported chapter formats:
        # - "Chapter 1", "Chapter1", "Ch1"
        # - "第1章", "第 1 章"
        # - "Chapter One", "第一章"
        # - "1" (simple number)
        
        text_lower = text.lower()
        pattern_lower = chapter_pattern.lower()
        
        # Extract number
        chapter_num_match = re.search(r'\d+', pattern_lower)
        if chapter_num_match:
            chapter_num = chapter_num_match.group()
            
            # Check various chapter formats
            patterns = [
                rf'chapter\s*{chapter_num}\b',
                rf'ch\s*{chapter_num}\b',
                rf'第\s*{chapter_num}\s*章',
                rf'\b{chapter_num}\s*章',
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        
        # Direct text matching
        if pattern_lower in text_lower:
            return True
        
        return False
    
    def _extract_chapter_from_filename(self, filename: str) -> Optional[str]:
        """Extract chapter information from filename"""
        # Match common chapter formats
        patterns = [
            r'chapter\s*(\d+)',
            r'ch\s*(\d+)',
            r'第\s*(\d+)\s*章',
            r'(\d+)\s*章',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"Chapter {match.group(1)}"
        
        return None
    
    def process_single_pdf(
        self, 
        pdf_path: str,
        user_requirements: Optional[str] = None,
        target_chapters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a single PDF file, optionally filtering content based on requirements

        Args:
            pdf_path: Path to the PDF file
            user_requirements: User requirements (for content filtering)
            target_chapters: List of target chapters

        Returns:
            Extracted slide data
        """
        pdf_file = Path(pdf_path)
        pdf_name = pdf_file.stem
        
        # Extract text content
        text_content = self.extract_text(pdf_path)
        
        # Identify slide structure
        slide_structure = self.identify_slide_structure(pdf_path, text_content)
        
        # If chapters are specified, filter relevant slides
        if target_chapters:
            slide_structure = self._filter_slides_by_chapters(
                slide_structure, 
                target_chapters
            )
        
        # If user requirements are provided, further filter relevant content
        if user_requirements:
            slide_structure = self._filter_slides_by_requirements(
                slide_structure,
                user_requirements
            )
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        return {
            "filename": pdf_file.name,
            "file_path": str(pdf_file),
            "pdf_name": pdf_name,
            "metadata": metadata,
            "total_pages": metadata.get("num_pages", 0),
            "extracted_slides_count": len(slide_structure),
            "text_content": text_content,
            "slide_structure": slide_structure,
            "processed_at": datetime.now().isoformat()
        }
    
    def _filter_slides_by_chapters(
        self,
        slide_structure: List[Dict[str, Any]],
        target_chapters: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter slides by chapter"""
        filtered = []
        
        for slide in slide_structure:
            slide_text = f"{slide.get('title', '')} {slide.get('content', '')}"
            
            # Check if it matches any target chapter
            if any(
                self._match_chapter_in_text(slide_text, chapter)
                for chapter in target_chapters
            ):
                filtered.append(slide)
        
        return filtered
    
    def _filter_slides_by_requirements(
        self,
        slide_structure: List[Dict[str, Any]],
        user_requirements: str
    ) -> List[Dict[str, Any]]:
        """Filter relevant content based on user requirements (can be extended to use embedding similarity)"""
        # Simple keyword matching can be used here, or embedding similarity
        # For simplicity, using keyword matching for now
        requirements_lower = user_requirements.lower()
        keywords = requirements_lower.split()
        
        filtered = []
        for slide in slide_structure:
            slide_text = f"{slide.get('title', '')} {slide.get('content', '')}".lower()
            
            # Check if it contains keywords
            if any(keyword in slide_text for keyword in keywords if len(keyword) > 2):
                filtered.append(slide)
        
        return filtered if filtered else slide_structure  # If no matches, return all
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text content from a PDF"""
        text_by_page = []
        
        if PDFPLUMBER_AVAILABLE:
            try:
                # Use pdfplumber for text extraction (more accurate)
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text:
                            text_by_page.append({
                                "page_number": page_num,
                                "text": text.strip(),
                                "char_count": len(text)
                            })
            except Exception as e:
                print(f"Warning: pdfplumber failed: {e}")
        
        if not text_by_page and PYPDF2_AVAILABLE:
            # Fallback: use PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            text_by_page.append({
                                "page_number": page_num,
                                "text": text.strip(),
                                "char_count": len(text)
                            })
            except Exception as e:
                print(f"Warning: PyPDF2 failed: {e}")
        
        if not text_by_page:
            raise ValueError(f"Could not extract text from {pdf_path}. Please install pdfplumber or PyPDF2.")
        
        full_text = "\n\n".join([page["text"] for page in text_by_page])
        
        return {
            "full_text": full_text,
            "pages": text_by_page,
            "total_characters": len(full_text)
        }
    
    def identify_slide_structure(self, pdf_path: str, text_content: Dict) -> List[Dict[str, Any]]:
        """Identify slide structure (titles, content, etc.)"""
        slides = []
        
        # Simple strategy: treat each page as one slide
        for page in text_content["pages"]:
            text = page["text"]
            
            # Try to identify title (usually the first line or largest font)
            lines = text.split("\n")
            title = lines[0].strip() if lines else "Untitled Slide"
            
            # Extract content
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
            
            slides.append({
                "slide_number": page["page_number"],
                "title": title[:100],  # Limit title length
                "content": content,
                "bullet_points": self.extract_bullet_points(content)
            })
        
        return slides
    
    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        lines = text.split("\n")
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Identify common bullet point markers
            if line and (line.startswith("•") or 
                        line.startswith("-") or 
                        line.startswith("*") or
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                bullet_points.append(line)
        
        return bullet_points
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = {
            "num_pages": 0,
            "file_size": os.path.getsize(pdf_path),
            "creation_date": None,
            "modification_date": None
        }
        
        if PYPDF2_AVAILABLE:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["num_pages"] = len(pdf_reader.pages)
                    
                    if pdf_reader.metadata:
                        metadata.update({
                            "title": pdf_reader.metadata.get("/Title"),
                            "author": pdf_reader.metadata.get("/Author"),
                            "subject": pdf_reader.metadata.get("/Subject"),
                            "creation_date": str(pdf_reader.metadata.get("/CreationDate")),
                            "modification_date": str(pdf_reader.metadata.get("/ModDate"))
                        })
            except Exception as e:
                print(f"Warning: Could not extract metadata: {e}")
        
        return metadata

