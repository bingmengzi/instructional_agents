"""
OptimizeSlidesDeliberation - Per-chapter slide optimization using the Deliberation pattern.

This module is the optimize-mode counterpart to SlidesDeliberation in slides.py.
It uses the same Agent + Deliberation abstractions from agents.py to orchestrate
multi-agent discussions for analyzing and improving existing slide content.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.agents import Agent, Deliberation, LLM
from src.slides import SlideUtils
from src.slide_knowledge_base import SlideKnowledgeBase


class OptimizeSlidesDeliberation:
    """
    Orchestrates per-chapter slide optimization using the Deliberation pattern.

    Two-phase deliberation:
      1. AnalysisDeliberation: Content Analyst + Improvement Advisor discuss existing slides
         -> Analysis Summarizer produces structured analysis and recommendations
      2. EnhancementDeliberation (per-slide): Content Enhancer + LaTeX Generator discuss
         how to improve each slide -> Enhancement Summarizer produces final LaTeX frames
    """

    def __init__(
        self,
        id: str,
        name: str,
        agents: Dict[str, Agent],
        llm: LLM,
        output_dir: str,
        knowledge_base: SlideKnowledgeBase,
    ):
        """
        Initialize OptimizeSlidesDeliberation.

        Args:
            id: Unique identifier (e.g. "optimize_chapter_1")
            name: Human-readable name (e.g. "Optimize - Chapter 1: Intro to ML")
            agents: Dictionary of agents keyed by role:
                - content_analyst, improvement_advisor, analysis_summarizer
                - content_enhancer, latex_generator, enhance_summarizer
            llm: LLM instance
            output_dir: Directory to save output files
            knowledge_base: Pre-loaded SlideKnowledgeBase for this chapter
        """
        self.id = id
        self.name = name
        self.agents = agents
        self.llm = llm
        self.output_dir = output_dir
        self.knowledge_base = knowledge_base

        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        chapter_slides: List[Dict[str, Any]],
        user_requirements: str,
        user_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the optimization deliberation for a chapter.

        Args:
            chapter_slides: List of slide dicts from the knowledge base
                (each has: title, content, slide_number, etc.)
            user_requirements: User's requirements for improvement
            user_feedback: Optional user feedback dict (e.g. {"slides": "...", "overall": "..."})

        Returns:
            Dict with success status, file paths, and statistics
        """
        if user_feedback is None:
            user_feedback = {"slides": "", "overall": ""}

        print(f"\n{'='*60}")
        print(f"Starting Optimize Deliberation: {self.name}")
        print(f"{'='*60}\n")
        print(f"Slides to optimize: {len(chapter_slides)}")
        print(f"User requirements: {user_requirements[:200]}...")

        total_time = 0
        total_tokens = 0

        # ── Phase 1: Analysis Deliberation ──────────────────────────
        print(f"\n{'#'*50}")
        print(f"Phase 1: Content Analysis & Recommendations")
        print(f"{'#'*50}\n")

        analysis_result, analysis_time, analysis_tokens = self._run_analysis_deliberation(
            chapter_slides, user_requirements
        )
        total_time += analysis_time
        total_tokens += analysis_tokens

        # Save analysis result
        analysis_path = os.path.join(self.output_dir, "analysis.md")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(f"# Content Analysis\n\n{analysis_result}")
        print(f"Analysis saved to: {analysis_path}")

        # ── Phase 2: Per-slide Enhancement Deliberation ─────────────
        print(f"\n{'#'*50}")
        print(f"Phase 2: Slide Enhancement & LaTeX Generation")
        print(f"{'#'*50}\n")

        # Get LaTeX template
        latex_template = SlideUtils.get_latex_template(catalog=False)
        latex_prefix, latex_suffix = SlideUtils.parse_latex_template(latex_template)

        enhanced_frames = []
        enhanced_content_list = []

        for idx, slide in enumerate(chapter_slides):
            slide_title = slide.get("title", f"Slide {idx + 1}")
            print(f"\n{'-'*50}")
            print(f"Enhancing Slide {idx + 1}/{len(chapter_slides)}: {slide_title}")
            print(f"{'-'*50}\n")

            # Search knowledge base for related content
            relevant_content = self.knowledge_base.search(slide_title, top_k=3)

            enhanced, frames, enh_time, enh_tokens = self._run_enhancement_deliberation(
                slide, idx, analysis_result, user_requirements, user_feedback, relevant_content
            )
            total_time += enh_time
            total_tokens += enh_tokens

            enhanced_content_list.append(enhanced)
            enhanced_frames.extend(frames)

        # ── Phase 3: Compile and Save ───────────────────────────────
        print(f"\n{'#'*50}")
        print(f"Phase 3: Compiling Results")
        print(f"{'#'*50}\n")

        # Compile full LaTeX document
        full_latex = SlideUtils.compile_latex_document(latex_prefix, enhanced_frames, latex_suffix)

        # Save LaTeX file
        latex_file = os.path.join(self.output_dir, "enhanced_slides.tex")
        with open(latex_file, "w", encoding="utf-8") as f:
            f.write(full_latex)

        # Save enhanced content summary
        content_file = os.path.join(self.output_dir, "enhanced_content.json")
        with open(content_file, "w", encoding="utf-8") as f:
            json.dump({
                "enhanced_at": datetime.now().isoformat(),
                "original_slides_count": len(chapter_slides),
                "enhanced_slides": enhanced_content_list,
                "user_requirements": user_requirements,
            }, f, indent=2, ensure_ascii=False)

        # Save statistics
        stats_file = os.path.join(self.output_dir, f"statistics_{self.id}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump({
                "elapsed_time": total_time,
                "token_usage": total_tokens,
                "total_slides": len(chapter_slides),
                "total_frames": len(enhanced_frames),
            }, f, indent=2)

        print(f"\nOptimize Deliberation Complete: {self.name}")
        print(f"  LaTeX: {latex_file}")
        print(f"  Content: {content_file}")
        print(f"  Frames generated: {len(enhanced_frames)}")
        print(f"  Time: {total_time:.2f}s | Tokens: {total_tokens}")

        return {
            "success": True,
            "latex_file": latex_file,
            "content_file": content_file,
            "analysis_file": analysis_path,
            "total_slides": len(chapter_slides),
            "total_frames": len(enhanced_frames),
            "elapsed_time": total_time,
            "token_usage": total_tokens,
        }

    def _run_analysis_deliberation(
        self,
        chapter_slides: List[Dict[str, Any]],
        user_requirements: str,
    ) -> tuple:
        """
        Run the analysis phase as a proper Deliberation.

        Content Analyst + Improvement Advisor discuss the existing slides,
        then the Analysis Summarizer produces a structured report.

        Returns:
            (analysis_text, elapsed_time, token_usage)
        """
        # Format existing slide content as context
        slides_summary = self._format_slides_for_context(chapter_slides)

        # Get knowledge base summary for additional context
        kb_summary = self.knowledge_base.get_all_content_summary()
        kb_info = f"Knowledge base contains {kb_summary.get('total_chunks', 0)} slide chunks."

        # Create the Analysis Deliberation using the standard Deliberation class
        analysis_deliberation = Deliberation(
            id=f"{self.id}_analysis",
            name=f"Content Analysis - {self.name}",
            agents=[
                self.agents["content_analyst"],
                self.agents["improvement_advisor"],
            ],
            summary_agent=self.agents["analysis_summarizer"],
            max_rounds=1,
            instruction_prompt=(
                "Analyze the existing slide content below. The Content Analyst should evaluate "
                "structure, coverage, quality, strengths, and weaknesses. The Improvement Advisor "
                "should then propose specific, actionable recommendations for enhancement.\n\n"
                f"Knowledge Base Info: {kb_info}"
            ),
            output_format="md",
        )

        # Run the deliberation
        result, elapsed_time, token_usage = analysis_deliberation.run(
            current_context=slides_summary,
            user_suggestion=user_requirements,
        )

        return result, elapsed_time, token_usage

    def _run_enhancement_deliberation(
        self,
        slide: Dict[str, Any],
        slide_idx: int,
        analysis_result: str,
        user_requirements: str,
        user_feedback: Dict[str, Any],
        relevant_content: List[Dict[str, Any]],
    ) -> tuple:
        """
        Run the enhancement phase for a single slide as a proper Deliberation.

        Content Enhancer + LaTeX Generator discuss the slide improvement,
        then the Enhancement Summarizer produces the final LaTeX frames.

        Returns:
            (enhanced_content_dict, latex_frames_list, elapsed_time, token_usage)
        """
        title = slide.get("title", f"Slide {slide_idx + 1}")
        content = slide.get("content", slide.get("text", ""))

        # Format relevant content from knowledge base search
        related_text = ""
        if relevant_content:
            related_items = []
            for i, item in enumerate(relevant_content[:3]):
                item_content = item.get("content", "")[:300]
                related_items.append(f"  Related {i+1}: {item_content}")
            related_text = "\n".join(related_items)

        # Format user feedback
        feedback_text = ""
        if user_feedback:
            slides_fb = user_feedback.get("slides", "")
            overall_fb = user_feedback.get("overall", "")
            if slides_fb:
                feedback_text += f"\nUser feedback on slides: {slides_fb}"
            if overall_fb:
                feedback_text += f"\nOverall user feedback: {overall_fb}"

        # Build the context for this slide's enhancement
        slide_context = (
            f"Original Slide Title: {title}\n"
            f"Original Content:\n{content[:2000]}\n\n"
            f"Analysis & Recommendations:\n{analysis_result[:2000]}\n"
        )
        if related_text:
            slide_context += f"\nRelated Content from Knowledge Base:\n{related_text}\n"
        if feedback_text:
            slide_context += f"\n{feedback_text}\n"

        # Create the Enhancement Deliberation
        enhancement_deliberation = Deliberation(
            id=f"{self.id}_enhance_{slide_idx}",
            name=f"Enhance Slide: {title}",
            agents=[
                self.agents["content_enhancer"],
                self.agents["latex_generator"],
            ],
            summary_agent=self.agents["enhance_summarizer"],
            max_rounds=1,
            instruction_prompt=(
                "Improve this slide based on the analysis and user requirements. "
                "The Content Enhancer should propose enhanced content that maintains the original "
                "structure while incorporating improvements. The LaTeX Generator should then discuss "
                "how to best present this content in beamer format.\n\n"
                "The final summary MUST contain the enhanced LaTeX frames using \\begin{frame}[fragile] ... \\end{frame} format. "
                "Generate at most 3 frames per slide. Follow these LaTeX guidelines:\n"
                "1. Don't use non-English characters directly (use LaTeX commands like $\\gamma$)\n"
                "2. Escape special characters (use \\& instead of &)\n"
                "3. Use itemize/enumerate for lists, block for highlights, lstlisting for code\n"
                "4. Keep each frame focused and not overcrowded"
            ),
            output_format="tex",
        )

        # Run the deliberation
        result, elapsed_time, token_usage = enhancement_deliberation.run(
            current_context=slide_context,
            user_suggestion=user_requirements,
        )

        # Extract LaTeX frames from the deliberation result
        frames = SlideUtils.extract_latex_frames(result)
        if not frames:
            # Fallback: create a simple frame from the result
            frames = [
                f"\\begin{{frame}}[fragile]\n"
                f"    \\frametitle{{{title}}}\n"
                f"    {content[:500]}\n"
                f"\\end{{frame}}"
            ]

        # Build enhanced content record
        enhanced_content = {
            "original_title": title,
            "enhanced_title": title,
            "original_content": content,
            "enhanced_content": result,
            "frames_count": len(frames),
        }

        return enhanced_content, frames, elapsed_time, token_usage

    def _format_slides_for_context(self, slides: List[Dict[str, Any]]) -> str:
        """Format slide list into a text summary for deliberation context."""
        lines = [f"Existing Slides ({len(slides)} total):\n"]
        for i, slide in enumerate(slides):
            title = slide.get("title", f"Slide {i+1}")
            content = slide.get("content", slide.get("text", ""))
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"--- Slide {i+1}: {title} ---")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)
