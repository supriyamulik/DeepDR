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
            "Microaneurysms only. This is the earliest stage of diabetic retinopathy, "
            "characterized by tiny balloon-like swelling in the retina's blood vessels."
        ),
        'Moderate Diabetic Retinopathy': (
            "More than just microaneurysms but less than Severe NPDR. Features may "
            "include dot and blot hemorrhages, hard exudates, and cotton wool spots. "
            "The blood vessels that nourish the retina may swell and distort."
        ),
        'Severe Diabetic Retinopathy': (
            "Severe nonproliferative diabetic retinopathy (NPDR). Characterized by any of the following: "
            "more than 20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in 2+ quadrants, "
            "or prominent intraretinal microvascular abnormalities (IRMA) in 1+ quadrant."
        ),
        'Proliferative Diabetic Retinopathy': (
            "Advanced stage with neovascularization (growth of new, fragile blood vessels) "
            "and/or vitreous/preretinal hemorrhage. High risk for severe vision loss. "
            "Urgent referral to a retinal specialist is highly recommended."
        )
    }
    
    severity = scan.final_diagnosis if scan.final_diagnosis else scan.ai_diagnosis
    description = dr_descriptions.get(severity, "Diagnosis pending or unrecognized stage.")
    
    template_path = 'report_pdf.html'
    context = {
        'scan': scan,
        'description': description
    }
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="DeepDR_Report_{scan.patient.patient_id}.pdf"'
    
    template = get_template(template_path)
    html = template.render(context)
    
    pisa_status = pisa.CreatePDF(html, dest=response, link_callback=link_callback)
    
    if pisa_status.err:
        return HttpResponse('We had some errors <pre>' + html + '</pre>')
    return response
