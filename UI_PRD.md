# Product Requirements Document (PRD)
## DeepDR - AI Diagnostic Console for Diabetic Retinopathy Detection

**Version:** 1.0  
**Last Updated:** April 28, 2026  
**Status:** Draft

---

## 1. Executive Summary

**DeepDR** is a lightweight web-based AI screening tool for pre-validating Diabetic Retinopathy (DR) in clinic settings. Clinic technicians upload retinal images, receive instant AI-generated severity classifications, and print results for the doctor's review before examination.

**Key Principle:** Quick pre-validation → Doctor confirms during examination (not a replacement diagnosis)

---

## 2. Product Overview

### 2.1 Product Vision
Enable ophthalmologists and healthcare professionals to quickly and accurately screen for diabetic retinopathy using AI-assisted image analysis with full transparency into model predictions.

### 2.2 Product Goals
- ✅ Reduce pre-screening time to <30 seconds per patient
- ✅ Provide clear severity classification (5 levels)
- ✅ Display 87%+ confidence in predictions
- ✅ Enable one-click print/export for doctor review
- ✅ Simple enough for minimal technical training
- ✅ Mobile-friendly for clinic environments

### 2.3 Key Features
1. Simple image upload (drag & drop)
2. One-click AI analysis
3. Severity classification with confidence
4. Segmentation reference maps
5. Print/PDF export for medical records
6. Mobile responsive interface

---

## 3. User Persona

### Primary User: Clinic Technician / Screening Nurse
- **Goal:** Quickly run pre-screening AI analysis on retinal images before doctor examination
- **Workflow:** Capture image → Run analysis → Get result → Print/save → Doctor reviews during examination
- **Tech Proficiency:** Low-Medium (minimal technical training needed)
- **Pain Points:** 
  - Time constraints in busy clinics
  - Need fast, reliable screening results
  - Must be easy to learn and use
  - Minimal data entry required
- **Needs:** 
  - Simple one-click upload
  - Clear severity indicator (pass/fail or risk level)
  - Confidence score for reference
  - Quick print/export
  - Minimal patient information required

---

## 4. Feature Requirements

### 4.1 Core Feature: Image Upload & Preview

#### 4.1.1 Upload Interface
- **Requirement:** User must be able to upload retinal images
- **Supported Formats:** JPG, PNG, TIFF
- **Max File Size:** 50 MB
- **Constraints:** 
  - Minimum resolution: 256x256px
  - Maximum resolution: 4096x4096px
- **Upload Methods:**
  - Drag & drop zone
  - File picker dialog

**UX Requirements:**
- Large, obvious upload area (dominant on page)
- Visual feedback on file selection
- File validation before upload
- Preview of selected image before analysis
- Clear, simple error messages
- Single-step process (no wizard needed)

---

#### 4.1.2 One-Click Analysis
- **Button:** "RUN ANALYSIS" - Large, primary button
- **Action:** Start automated classification + segmentation
- **Feedback:** Loading animation with progress indicator
- **Processing Time Target:** <15 seconds total
- **Error Handling:** Clear retry button on failure

**UX Requirements:**
- Single button click to start (no configuration)
- Loading spinner with text (e.g., "Processing... 60%")
- Cannot click button again until analysis completes
- Optional: Cancel button during processing

---

### 4.2 Core Feature: Results Display

#### 4.2.1 Primary Diagnosis Card
Display the AI-generated diagnosis prominently:

```
┌────────────────────────────────────────────┐
│ SCREENING RESULT                           │
├────────────────────────────────────────────┤
│                                            │
│ Classification: MODERATE DR                │
│ Confidence: 87%                            │
│ Risk Level: [●●●○○] HIGH RISK             │
│                                            │
│ Severity Track:                            │
│ No DR ─ Mild ─ Moderate ─ Severe ─ PDR   │
│            ├────────●                      │
│                                            │
│ Recommendation:                            │
│ Doctor examination advised within 2 weeks  │
│                                            │
└────────────────────────────────────────────┘
```

**Data Elements:**
- DR Classification (text: "No DR", "Mild DR", etc.)
- Confidence percentage (0-100%)
- Risk badge with visual indicator
- Severity scale visualization
- Simple recommendation text

**Color Coding:**
- No DR: Green
- Mild: Yellow
- Moderate: Orange
- Severe: Red
- Proliferative: Dark Red

---

#### 4.2.2 Segmentation Reference Maps (Optional)
Display 6 segmentation masks for reference during doctor examination:

```
SEGMENTATION MAPS (for reference):

┌──────────────┬──────────────┬──────────────┐
│ Blood Vessel │ Hemorrhage   │ Hard Exudate │
│              │              │              │
│  [MAP]       │  [MAP]       │  [MAP]       │
└──────────────┴──────────────┴──────────────┘
┌──────────────┬──────────────┬──────────────┐
│ Microaneury  │ Optical Disc │ Soft Exudate │
│              │              │              │
│  [MAP]       │  [MAP]       │  [MAP]       │
└──────────────┴──────────────┴──────────────┘
```

**Requirements:**
- Show all 6 maps in compact 3x2 grid
- Toggleable (show/hide to save space)
- Click to enlarge individual maps
- Labels clearly identifying each abnormality

---

### 4.3 Core Feature: Patient Information (Optional/Minimal)

For pre-screening workflow, patient info is minimal:

```
┌────────────────────────────────────────┐
│ PATIENT INFORMATION (Optional)          │
├────────────────────────────────────────┤
│ Patient ID: [Optional field]            │
│ Eye: [Left / Right] dropdown            │
│ Notes: [Optional text area]             │
│                                         │
│ [Save to File] [Clear]                  │
└────────────────────────────────────────┘
```

**Requirements:**
- Minimal fields (optional data entry)
- Patient ID - for linking to records
- Eye selection - Left/Right
- Notes - for technician notes (optional)
- **NOT required** for analysis to proceed
- Auto-timestamp for all results

---

### 4.4 Core Feature: Quick Export/Print

#### 4.4.1 Action Buttons (After Analysis)
```
┌────────────────────────────────────────┐
│ [PRINT REPORT]  [SAVE PDF]  [NEW TEST]  │
└────────────────────────────────────────┘
```

**Print Report:**
- Opens browser print dialog
- Shows: Original image, classification, confidence, segmentation maps
- Patient info (if entered)
- Timestamp
- Simple, professional layout

**Save PDF:**
- One-click PDF download
- Filename format: `DeepDR_Result_[Date]_[Time].pdf`
- Same content as printed report

**New Test:**
- Resets UI for next patient
- Clears image preview
- Optional patient info clears

---

### 4.5 Core Feature: Navigation (Minimal)

Simple header only:

```
┌──────────────────────────────────────────┐
│ DeepDR Logo │ RETINAL SCREENING │ Help   │
└──────────────────────────────────────────┘
```

**Navigation Items:**
- Logo (home/reset)
- App title
- Help link (documentation)
- **NO** patient history, settings, or admin functions

---

## 5. User Flow

### 5.1 Main Screening Workflow

```
START
  ↓
1. Technician opens DeepDR UI
  ↓
2. (Optional) Enter Patient ID and eye side
  ↓
3. Click upload zone / Drag image into zone
  ├─ File validation
  ├─ Preview display
  └─ Confirm looks correct
  ↓
4. Click "RUN ANALYSIS"
  ├─ Show loading animation
  ├─ Process classification (CNN)
  ├─ Process 6 segmentations (U-Net)
  └─ Update progress indicator
  ↓
5. Results Display
  ├─ Severity classification (No/Mild/Moderate/Severe/PDR)
  ├─ Confidence score (%)
  ├─ Risk badge (color-coded)
  ├─ Optional: Segmentation maps
  └─ Enable export buttons
  ↓
6. Technician Reviews
  ├─ Checks severity level
  ├─ Notes confidence score
  └─ Reviews segmentation maps (optional)
  ↓
7. Export Result
  ├─ [PRINT REPORT] → Print for doctor
  ├─ [SAVE PDF] → Save to patient file
  └─ [NEW TEST] → Reset for next patient
  ↓
8. Patient proceeds to doctor with printed result
  ↓
END (Doctor confirms/adjusts during examination)
```

---

### 5.2 Error Scenarios

**Invalid File:**
```
User uploads non-image file
  ↓
Error: "Please upload a valid image file (JPG, PNG, TIFF)"
  ↓
User can retry with correct file
```

**Analysis Failure:**
```
Processing fails / Model error
  ↓
Error: "Analysis failed. Please try again."
  ↓
[RETRY] button to re-run
```

---

## 6. Layout & Wireframes

### 6.1 Desktop Layout (Recommended: 1024px+)

```
┌────────────────────────────────────────────────────────────────┐
│  DeepDR Logo │ RETINAL SCREENING │ Help                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐     ┌────────────────────────────────┐  │
│  │  UPLOAD ZONE     │     │  RESULT AREA (Hidden until    │  │
│  │                  │     │  analysis complete)           │  │
│  │  [Drop image     │     │                               │  │
│  │   or click]      │     │  ┌──────────────────────────┐ │  │
│  │                  │     │  │ CLASSIFICATION: MOD DR   │ │  │
│  │  📁 Choose File  │     │  │ Confidence: 87%          │ │  │
│  │                  │     │  │ Risk: ●●●○○ HIGH        │ │  │
│  │ [Preview Image]  │     │  │                          │ │  │
│  │                  │     │  │ Scale:                   │ │  │
│  └──────────────────┘     │  │ No - Mild - Mod - Sev-PDR│ │  │
│                           │  │      ├─────●             │ │  │
│  ┌──────────────────┐     │  └──────────────────────────┘ │  │
│  │ PATIENT INFO     │     │                               │  │
│  │ (Optional)       │     │  [Show Maps] [Hide Maps]      │  │
│  │                  │     │                               │  │
│  │ ID: [    ]       │     │  SEGMENTATION MAPS:           │  │
│  │ Eye: [Left ▼]    │     │  ┌────┬────┬────┐            │  │
│  │ Notes: [      ]  │     │  │BV  │Hem │HE  │            │  │
│  │                  │     │  ├────┼────┼────┤            │  │
│  │                  │     │  │Ma  │OD  │SE  │            │  │
│  │                  │     │  └────┴────┴────┘            │  │
│  └──────────────────┘     │                               │  │
│                           │  [PRINT] [SAVE PDF] [NEW]    │  │
│  ┌──────────────────┐     │                               │  │
│  │ [RUN ANALYSIS]   │     │                               │  │
│  │ (Primary Button) │     │                               │  │
│  └──────────────────┘     │                               │  │
│                           │                               │  │
└────────────────────────────────────────────────────────────────┘
```

**Left Sidebar:**
- Upload zone (dominant)
- Patient info (minimal, optional)
- RUN ANALYSIS button
- Simple, focused

**Right Content Area:**
- Empty state initially
- Results appear here after analysis
- Segmentation maps toggleable
- Export buttons at bottom

---

### 6.2 Mobile Layout (< 768px)

```
┌──────────────────────────────┐
│ DeepDR │ SCREENING │ Help     │
├──────────────────────────────┤
│                              │
│  [TAB: UPLOAD] [TAB: RESULT] │
│                              │
├──────────────────────────────┤
│ UPLOAD TAB (Active)          │
│                              │
│ UPLOAD ZONE                  │
│ ┌──────────────────────────┐ │
│ │                          │ │
│ │   📁 Drop or Click       │ │
│ │                          │ │
│ └──────────────────────────┘ │
│                              │
│ PATIENT ID: [Optional]       │
│ EYE: [Left ▼]                │
│ NOTES: [Optional]            │
│                              │
│ ┌──────────────────────────┐ │
│ │  [RUN ANALYSIS]          │ │
│ └──────────────────────────┘ │
│                              │
└──────────────────────────────┘

RESULT TAB (After Analysis):

┌──────────────────────────────┐
│ DeepDR │ SCREENING │ Help     │
├──────────────────────────────┤
│                              │
│  [TAB: UPLOAD] [TAB: RESULT] │
│                              │
├──────────────────────────────┤
│ RESULT TAB (Active)          │
│                              │
│ CLASSIFICATION               │
│ MODERATE DR                  │
│ Confidence: 87%              │
│ Risk: ●●●○○ HIGH            │
│                              │
│ Scale:                       │
│ No-Mild-Mod-Sev-PDR         │
│      ├─●                     │
│                              │
│ [SHOW MAPS]                  │
│                              │
│ [PRINT]                      │
│ [SAVE PDF]                   │
│ [NEW TEST]                   │
│                              │
└──────────────────────────────┘
```

---

## 7. Design Specifications

### 7.1 Color Palette

| Element | Color | Hex Code | Usage |
|---------|-------|----------|-------|
| Primary | Medical Blue | #1E40AF | Buttons, links, active states |
| Success | Green | #10B981 | Normal/No DR status |
| Warning | Amber | #F59E0B | Mild/Moderate alerts |
| Danger | Red | #EF4444 | Severe/Critical alerts |
| Critical | Dark Red | #991B1B | Proliferative DR |
| Gray | Neutral | #6B7280 | Text, borders, secondary |
| Background | Off-White | #F9FAFB | Page background |
| Card | White | #FFFFFF | Cards, panels |
| Border | Light Gray | #E5E7EB | Dividers, borders |

---

### 7.2 Typography

**Font Family:** 'Inter', 'Segoe UI', sans-serif

| Element | Size | Weight | Line Height |
|---------|------|--------|-------------|
| Page Title (H1) | 32px | 700 | 1.2 |
| Section Title (H2) | 24px | 600 | 1.3 |
| Card Title (H3) | 18px | 600 | 1.4 |
| Body Text | 14px | 400 | 1.6 |
| Small Text | 12px | 400 | 1.5 |
| Button Text | 14px | 600 | 1 |
| Medical Term | 14px | 500 | 1.6 |

---

### 7.3 Spacing System

**Base Unit:** 4px (Multiples of 4)

| Use Case | Size |
|----------|------|
| Extra Small | 4px |
| Small | 8px |
| Normal | 12px |
| Medium | 16px |
| Large | 24px |
| Extra Large | 32px |
| Huge | 48px |

**Common Patterns:**
- Card padding: 24px
- Section margins: 32px
- Button padding: 12px 24px
- Input field height: 40px

---

### 7.4 Component Specifications

#### 7.4.1 Buttons

**Primary Button:**
- Background: #1E40AF
- Text: White
- Padding: 12px 24px
- Border-radius: 6px
- Hover: #1e3a8a (darker)
- Active: #1e3a8a
- Disabled: #D1D5DB (gray)
- Font: 14px, 600 weight

**Secondary Button:**
- Background: #E5E7EB
- Text: #1F2937
- Padding: 12px 24px
- Border-radius: 6px
- Hover: #D1D5DB

**Danger Button:**
- Background: #EF4444
- Text: White
- Padding: 12px 24px

---

#### 7.4.2 Cards

- Background: #FFFFFF
- Border: 1px solid #E5E7EB
- Border-radius: 8px
- Box-shadow: 0 1px 3px rgba(0,0,0,0.1)
- Padding: 24px
- Margin-bottom: 24px

---

#### 7.4.3 Input Fields

- Background: #FFFFFF
- Border: 1px solid #E5E7EB
- Border-radius: 6px
- Padding: 10px 12px
- Font-size: 14px
- Focus: Border color #1E40AF, Box-shadow: 0 0 0 3px rgba(30,64,175,0.1)
- Placeholder: #9CA3AF

---

#### 7.4.4 Upload Zone

- Background: #F3F4F6
- Border: 2px dashed #1E40AF
- Border-radius: 8px
- Padding: 48px 24px
- Min-height: 200px
- Hover: Background #E0E7FF
- Active (drag): Background #DBEAFE, Border solid

---

### 7.5 Status Indicators

**DR Severity Badges:**

```
┌─────────────────────────────────────┐
│ ✓ No DR         │ Green  │ #10B981 │
│ ⚠ Mild DR       │ Yellow │ #F59E0B │
│ ⚠ Moderate DR   │ Orange │ #EA580C │
│ ✗ Severe DR     │ Red    │ #EF4444 │
│ ✗ Proliferative │ Dark   │ #991B1B │
└─────────────────────────────────────┘
```

---

## 8. Technical Requirements

### 8.1 Frontend Stack
- **HTML5** - Semantic markup
- **CSS3** - Grid, Flexbox, CSS Variables
- **JavaScript (ES6+)** - Vanilla JS or minimal framework
- **Libraries:**
  - Alpine.js or similar (lightweight interactivity)
  - Fetch API for AJAX
  - Canvas API for image manipulation
  - Optional: Chart.js for confidence visualization

### 8.2 Backend Requirements (Flask)
- Image upload handling
- Classification prediction endpoint
- Segmentation processing endpoint
- Patient record storage
- PDF report generation (weasyprint/reportlab)
- File management system

### 8.3 Performance Targets
- Page load: <2 seconds
- Image upload: <5 seconds for 5MB
- Classification: <3 seconds
- Segmentation (per module): <2 seconds
- PDF generation: <5 seconds
- Report display: <1 second

### 8.4 Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile: iOS Safari 14+, Chrome Mobile 90+

### 8.5 Accessibility
- WCAG 2.1 AA compliance
- Keyboard navigation
- Screen reader support
- Color contrast ratio 4.5:1 minimum
- Alt text for all images
- Semantic HTML structure

---

## 9. File Structure

```
Classification/
├── templates/
│   ├── base.html          # Base layout template
│   └── index.html         # Main screening interface (redesigned)
├── static/
│   ├── css/
│   │   ├── style.css      # Main styles (redesigned)
│   │   └── responsive.css # Mobile/tablet styles
│   ├── js/
│   │   ├── main.js        # Main logic
│   │   └── upload.js      # Upload & analysis handling
│   └── images/
│       └── logo.svg
├── model/
│   └── model.h5           # Classification model (to be downloaded)
├── app.py                 # Flask app (updated)
└── requirements.txt       # Dependencies
```

---

## 10. Implementation Roadmap

### Phase 1: MVP Core (Week 1-2)
- [ ] Redesign HTML structure (semantic layout)
- [ ] Implement CSS styling per design specs
- [ ] Image upload with drag & drop
- [ ] Single button analysis
- [ ] Results display (classification + confidence)
- [ ] Print/PDF export functionality
- [ ] Mobile responsive layout

### Phase 2: Enhancement (Week 3)
- [ ] Segmentation maps display
- [ ] Toggle show/hide maps
- [ ] Improved error handling
- [ ] Loading states with progress

### Phase 3: Polish (Week 4)
- [ ] Cross-browser testing
- [ ] Performance optimization
- [ ] Accessibility improvements
- [ ] Optional: Camera capture input

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|------------|
| Page Load Time | <2s | Lighthouse, Chrome DevTools |
| Analysis Time | <15s total | End-to-end timing |
| User Satisfaction | >4/5 | Post-analysis survey |
| Error Rate | <1% | System logs |
| Mobile Usability | >90% | Mobile usability test |
| Accessibility Score | >95 | axe DevTools |
| Browser Compatibility | 100% | Cross-browser testing |

---

## 12. Future Enhancements (Post-MVP)

- [ ] Camera capture directly from web
- [ ] Batch processing for multiple images
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Local database for storing results
- [ ] QR code generation for result sharing
- [ ] Integration with hospital EHR systems
- [ ] Explainable AI (Grad-CAM visualization)
- [ ] Similar case comparison
- [ ] Performance metrics dashboard

---

## 13. Appendix: Key Definitions

- **DR Level 0:** No Diabetic Retinopathy
- **DR Level 1:** Mild (microaneurisms only)
- **DR Level 2:** Moderate (microaneurisms + other abnormalities)
- **DR Level 3:** Severe (extensive hemorrhaging/exudates)
- **DR Level 4:** Proliferative (neovascularization)

---

**End of PRD**

For questions or clarifications, contact: [Development Team]
