import os
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.template.loader import get_template
from xhtml2pdf import pisa
from .models import Patient, ScanSession
from analyzer.ml_pipeline import ModelManager

# Lazy load the AI model stack
def get_model_manager():
    manager = ModelManager()
    if not manager.initialized:
        manager.initialize()
    return manager

def dashboard(request):
    scans = ScanSession.objects.all().order_by('-created_at')
    
    # Calculate quick stats
    total_scans = scans.count()
    pending_review = scans.filter(status='PENDING').count()
    referred_cases = scans.filter(status='REFERRED').count()
    completed_cases = scans.filter(status='COMPLETED').count()
    
    context = {
        'scans': scans,
        'total_scans': total_scans,
        'pending_review': pending_review,
        'referred_cases': referred_cases,
        'completed_cases': completed_cases,
        'active_tab': 'dashboard'
    }
    return render(request, 'dashboard.html', context)

def upload_scan(request):
    if request.method == 'POST':
        # Retrieve form data
        patient_id = request.POST.get('patient_id', '').strip()
        name = request.POST.get('name', '').strip()
        age = request.POST.get('age', '').strip()
        gender = request.POST.get('gender', '').strip()
        original_image = request.FILES.get('original_image')
        
        if not (patient_id and name and age and gender and original_image):
            messages.error(request, "Please fill out all patient fields and upload an image.")
            return render(request, 'upload.html', {'active_tab': 'upload'})
            
        try:
            # 1. Get or create patient record
            patient, created = Patient.objects.get_or_create(
                patient_id=patient_id,
                defaults={
                    'name': name,
                    'age': int(age),
                    'gender': gender
                }
            )
            if not created:
                # Update info if patient already existed
                patient.name = name
                patient.age = int(age)
                patient.gender = gender
                patient.save()
                
            # 2. Save initial scan session (to write file to disk)
            scan_session = ScanSession.objects.create(
                patient=patient,
                original_image=original_image,
                status='PENDING'
            )
            
            # 3. Execute AI analysis pipeline
            manager = get_model_manager()
            results = manager.run_full_pipeline(
                scan_session.original_image.path, 
                os.path.basename(scan_session.original_image.name)
            )
            
            # 4. Save analysis results to database
            scan_session.ai_diagnosis = results["diagnosis"]
            scan_session.ai_severity_index = results["severity_index"]
            scan_session.confidence_breakdown = results["confidence_breakdown"]
            
            scan_session.mask_blood_vessel = results["masks"]["blood_vessel"]
            scan_session.mask_hemorrhage = results["masks"]["hemorrhage"]
            scan_session.mask_hard_exudate = results["masks"]["hard_exudate"]
            scan_session.mask_microaneurysm = results["masks"]["microaneurysm"]
            scan_session.mask_optic_disc = results["masks"]["optic_disc"]
            scan_session.mask_soft_exudate = results["masks"]["soft_exudate"]
            
            scan_session.combined_overlay = results["combined_overlay"]
            scan_session.save()
            
            messages.success(request, f"Scan uploaded and analyzed successfully for patient {name}!")
            return redirect('scan_results', scan_id=scan_session.id)
            
        except Exception as e:
            messages.error(request, f"Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return render(request, 'upload.html', {'active_tab': 'upload'})
            
    return render(request, 'upload.html', {'active_tab': 'upload'})

def scan_results(request, scan_id):
    scan = get_object_or_404(ScanSession, id=scan_id)
    
    if request.method == 'POST':
        doctor_notes = request.POST.get('doctor_notes', '').strip()
        final_diagnosis = request.POST.get('final_diagnosis', '')
        status = request.POST.get('status', 'PENDING')
        
        # DR levels mapping to severity index
        DR_MAP = {
            'No Diabetic Retinopathy': 0,
            'Mild Diabetic Retinopathy': 1,
            'Moderate Diabetic Retinopathy': 2,
            'Severe Diabetic Retinopathy': 3,
            'Proliferative Diabetic Retinopathy': 4
        }
        
        scan.doctor_notes = doctor_notes
        scan.final_diagnosis = final_diagnosis
        scan.final_severity_index = DR_MAP.get(final_diagnosis, scan.ai_severity_index)
        scan.status = status
        scan.save()
        
        messages.success(request, "Clinical diagnosis signature updated successfully.")
        return redirect('scan_results', scan_id=scan.id)
        
    # Render detail page
    return render(request, 'results.html', {'scan': scan, 'active_tab': 'dashboard'})

def grading_queue(request):
    scans = ScanSession.objects.filter(status='PENDING').order_by('-created_at')
    return render(request, 'grading_queue.html', {'scans': scans, 'active_tab': 'grading'})

def referrals_queue(request):
    scans = ScanSession.objects.filter(status='REFERRED').order_by('-created_at')
    return render(request, 'referrals.html', {'scans': scans, 'active_tab': 'referrals'})

def explainability(request, scan_id):
    scan = get_object_or_404(ScanSession, id=scan_id)
    manager = get_model_manager()
    
    # Check if we already generated explainability data or need to generate it
    if scan.explainability_data:
        # Load from cache
        gradcam_comparison = scan.explainability_data.get('gradcam_comparison')
        gradcam_seg_overlays = scan.explainability_data.get('gradcam_seg_overlays')
        clinical_summary = scan.explainability_data.get('clinical_summary')
    else:
        # Generate dynamically
        seg_mask_urls = {
            "blood_vessel": scan.mask_blood_vessel,
            "hemorrhage": scan.mask_hemorrhage,
            "hard_exudate": scan.mask_hard_exudate,
            "microaneurysm": scan.mask_microaneurysm,
            "optic_disc": scan.mask_optic_disc,
            "soft_exudate": scan.mask_soft_exudate
        }
        
        explainability_data = manager.run_explainability(
            scan.original_image.path,
            os.path.basename(scan.original_image.name),
            seg_mask_urls
        )
        
        gradcam_comparison = explainability_data['gradcam_comparison']
        gradcam_seg_overlays = explainability_data['gradcam_seg_overlays']
        
        # Generate Clinical Summary via Groq
        from dotenv import load_dotenv
        import groq
        
        load_dotenv(os.path.join(settings.BASE_DIR, '.env'), override=True)
        api_key = os.getenv("GROQ_API_KEY")
        
        clinical_summary = ""
        if not api_key or api_key.strip() == "" or api_key == "your_api_key_here":
            clinical_summary = "<p style='color: #b45309;'>⚠️ Groq API Key not configured. Add your key to the <code>.env</code> file to enable AI clinical summaries.</p>"
        else:
            try:
                groq_client = groq.Groq(api_key=api_key)
                confidence_val = scan.confidence_breakdown['probs'][scan.ai_severity_index] if scan.confidence_breakdown and 'probs' in scan.confidence_breakdown else 0
                prompt = f"""You are a gentle, supportive clinical AI assistant working alongside an ophthalmologist. A patient has received an AI-generated retinal scan result.

Diagnosis: {scan.ai_diagnosis}
AI Confidence Score: {confidence_val}%

Write a calm, medical-friendly patient explanation in exactly 3 short paragraphs. The tone should be reassuring but clinical, without sounding overly robotic, strict, or aggressive. Label each paragraph with a heading using this exact format:
**What This Means**
[Explain the diagnosis simply and calmly. If it's early stage, reassure them it's manageable. Do not sound alarming or definitive.]

**Understanding the Confidence Score**
[Explain the {confidence_val}% confidence score simply. Frame it as the AI's internal certainty level based on visual patterns, and remind them that AI is just a screening tool. Keep it brief.]

**Important Disclaimer**
[One brief, gentle sentence reminding them that this AI tool is meant to assist their doctor, and their ophthalmologist will review these results to provide the final medical diagnosis.]

Be concise, warm, and professional. No markdown lists, no bullet points, just paragraph text under bold headings."""

                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=350
                )
                raw_summary = completion.choices[0].message.content
                import re
                # Convert **Heading** to styled heading divs
                formatted = re.sub(
                    r'\*\*(.*?)\*\*',
                    r'<div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--color-ink-mute); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1.25rem; margin-bottom: 0.4rem;">\1</div>',
                    raw_summary
                )
                # Convert double newlines to paragraph breaks
                formatted = re.sub(r'\n\n+', '</p><p style="margin-bottom: 0;">', formatted)
                formatted = re.sub(r'\n', ' ', formatted)
                clinical_summary = f'<p style="margin-bottom: 0;">{formatted}</p>'
            except Exception as e:
                clinical_summary = f"<p style='color: #b91c1c;'>⚠️ Failed to generate AI summary: {e}</p>"
        
        # Save to DB cache
        scan.explainability_data = {
            'gradcam_comparison': gradcam_comparison,
            'gradcam_seg_overlays': gradcam_seg_overlays,
            'clinical_summary': clinical_summary
        }
        scan.save()
            
    context = {
        'scan': scan,
        'gradcam_comparison': gradcam_comparison,
        'gradcam_seg_overlays': gradcam_seg_overlays,
        'clinical_summary': clinical_summary,
        'active_tab': 'dashboard',  # Or keep dashboard active for inner views
        'is_explainability': True
    }
    
    return render(request, 'explainability.html', context)

def link_callback(uri, rel):
    """
    Convert HTML URIs to absolute system paths so xhtml2pdf can access those
    resources (images/CSS) from disk.
    """
    sUrl = settings.STATIC_URL
    sRoot = settings.STATIC_ROOT if hasattr(settings, 'STATIC_ROOT') and settings.STATIC_ROOT else settings.STATICFILES_DIRS[0]
    mUrl = settings.MEDIA_URL
    mRoot = settings.MEDIA_ROOT

    if uri.startswith(mUrl):
        path = os.path.join(mRoot, uri.replace(mUrl, ""))
    elif uri.startswith(sUrl):
        path = os.path.join(sRoot, uri.replace(sUrl, ""))
    else:
        return uri

    if not os.path.isfile(path):
        print(f"Warning: PDF image not found at {path}")
    return path

def generate_report_pdf(request, scan_id):
    scan = get_object_or_404(ScanSession, id=scan_id)
    
    dr_descriptions = {
        'No Diabetic Retinopathy': (
            "No abnormalities characteristic of diabetic retinopathy are present. "
            "No microaneurysms or other lesions detected."
        ),
        'Mild Diabetic Retinopathy': (
            "Signs of Mild NPDR with microaneurysms only. This is the earliest stage of diabetic retinopathy, "
            "characterized by tiny balloon-like swelling in the retina's blood vessels."
        ),
        'Moderate Diabetic Retinopathy': (
            "Signs of Moderate NPDR with dot and blot hemorrhages, hard exudates, and/or cotton wool spots detected. "
            "The blood vessels that nourish the retina may swell and distort."
        ),
        'Severe Diabetic Retinopathy': (
            "Severe NPDR detected. Characterized by extensive intraretinal hemorrhages in multiple quadrants, "
            "definite venous beading, and/or prominent intraretinal microvascular abnormalities (IRMA)."
        ),
        'Proliferative Diabetic Retinopathy': (
            "Proliferative DR detected with neovascularization (growth of new, fragile blood vessels) "
            "and/or vitreous/preretinal hemorrhage. High risk for severe vision loss."
        )
    }

    dr_icd_codes = {
        'No Diabetic Retinopathy': ('E11.319', 'Type 2 diabetes mellitus without diabetic retinopathy'),
        'Mild Diabetic Retinopathy': ('E11.321', 'Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema'),
        'Moderate Diabetic Retinopathy': ('E11.331', 'Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy with macular edema'),
        'Severe Diabetic Retinopathy': ('E11.341', 'Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy'),
        'Proliferative Diabetic Retinopathy': ('E11.351', 'Type 2 diabetes mellitus with proliferative diabetic retinopathy')
    }

    dr_screening_result = {
        'No Diabetic Retinopathy': 'Negative for diabetic retinopathy.',
        'Mild Diabetic Retinopathy': 'Positive for mild nonproliferative diabetic retinopathy.',
        'Moderate Diabetic Retinopathy': 'Positive for vision threatening diabetic retinopathy.',
        'Severe Diabetic Retinopathy': 'Positive for vision threatening diabetic retinopathy.',
        'Proliferative Diabetic Retinopathy': 'Positive for vision threatening proliferative diabetic retinopathy.'
    }

    dr_recommendations = {
        'No Diabetic Retinopathy': (
            'Routine follow-up with an ophthalmologist is recommended within 12 months. '
            'As per ADA recommendations, emphasize the importance of controlling blood sugar, '
            'cholesterol and blood pressure as well as the importance of routine follow-up with '
            'an ophthalmologist regardless of whether visual symptoms are present or absent.'
        ),
        'Mild Diabetic Retinopathy': (
            'Follow-up with an ophthalmologist within 6-9 months is recommended. '
            'As per ADA recommendations, emphasize the importance of controlling blood sugar, '
            'cholesterol and blood pressure as well as the importance of routine follow-up with '
            'an ophthalmologist regardless of whether visual symptoms are present or absent.'
        ),
        'Moderate Diabetic Retinopathy': (
            'Referral to an ophthalmologist for evaluation of vision threatening diabetic retinopathy '
            'is recommended within 2-4 weeks. '
            'As per ADA recommendations, emphasize the importance of controlling blood sugar, '
            'cholesterol and blood pressure as well as the importance of routine follow-up with '
            'an ophthalmologist regardless of whether visual symptoms are present or absent.'
        ),
        'Severe Diabetic Retinopathy': (
            'Immediate referral to a retinal specialist for evaluation of vision threatening diabetic '
            'retinopathy is strongly recommended. '
            'As per ADA recommendations, emphasize the importance of controlling blood sugar, '
            'cholesterol and blood pressure as well as the importance of routine follow-up with '
            'an ophthalmologist regardless of whether visual symptoms are present or absent.'
        ),
        'Proliferative Diabetic Retinopathy': (
            'URGENT: Immediate referral to a retinal specialist for evaluation and potential intervention '
            '(pan-retinal photocoagulation or anti-VEGF therapy) is strongly recommended. '
            'As per ADA recommendations, emphasize the importance of controlling blood sugar, '
            'cholesterol and blood pressure as well as the importance of routine follow-up with '
            'an ophthalmologist regardless of whether visual symptoms are present or absent.'
        )
    }
    
    severity = scan.final_diagnosis if scan.final_diagnosis else scan.ai_diagnosis
    description = dr_descriptions.get(severity, "Diagnosis pending or unrecognized stage.")
    icd_code, icd_desc = dr_icd_codes.get(severity, ('—', 'Unrecognized stage'))
    screening_result = dr_screening_result.get(severity, 'Screening result pending.')
    recommendation = dr_recommendations.get(severity, 'Follow up with your ophthalmologist.')
    
    template_path = 'report_pdf.html'
    
    clinical_summary = None
    gradcam_url = None
    if scan.explainability_data:
        clinical_summary = scan.explainability_data.get('clinical_summary')
        gradcam_comparison = scan.explainability_data.get('gradcam_comparison')
        if gradcam_comparison:
            best_layer = gradcam_comparison.get('best_layer')
            best_method = gradcam_comparison.get('best_method')
            for layer in gradcam_comparison.get('layers', []):
                if layer.get('name') == best_layer:
                    methods = layer.get('methods', {})
                    if best_method in methods:
                        gradcam_url = methods[best_method].get('overlay_url')

    from datetime import datetime
    now = datetime.now()
                        
    context = {
        'scan': scan,
        'description': description,
        'clinical_summary': clinical_summary,
        'gradcam_url': gradcam_url,
        'icd_code': icd_code,
        'icd_desc': icd_desc,
        'screening_result': screening_result,
        'recommendation': recommendation,
        'report_date': now.strftime('%Y-%b-%d %H:%M'),
    }
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="DeepDR_Report_{scan.patient.patient_id}.pdf"'
    
    template = get_template(template_path)
    html = template.render(context)
    
    pisa_status = pisa.CreatePDF(html, dest=response, link_callback=link_callback)
    
    if pisa_status.err:
        return HttpResponse('We had some errors <pre>' + html + '</pre>')
    return response
