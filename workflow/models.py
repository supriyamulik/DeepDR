from django.db import models

class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    patient_id = models.CharField(max_length=50, unique=True, verbose_name="Patient ID")
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.patient_id})"

class ScanSession(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending Doctor Review'),
        ('REFERRED', 'Referred to Specialist'),
        ('COMPLETED', 'Completed'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='scans')
    original_image = models.ImageField(upload_to='scans/original/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # AI Classification Results
    ai_diagnosis = models.CharField(max_length=100, blank=True, null=True)
    ai_severity_index = models.IntegerField(default=0)
    # Store confidence scores or probabilities as JSON (Feature 3 breakdown)
    confidence_breakdown = models.JSONField(blank=True, null=True)
    
    # AI Segmentation Mask Paths (Feature 5 outputs relative to media root)
    mask_blood_vessel = models.CharField(max_length=255, blank=True, null=True)
    mask_hemorrhage = models.CharField(max_length=255, blank=True, null=True)
    mask_hard_exudate = models.CharField(max_length=255, blank=True, null=True)
    mask_microaneurysm = models.CharField(max_length=255, blank=True, null=True)
    mask_optic_disc = models.CharField(max_length=255, blank=True, null=True)
    mask_soft_exudate = models.CharField(max_length=255, blank=True, null=True)
    
    # Combined OpenCV overlay image (Feature 5 final image)
    combined_overlay = models.ImageField(upload_to='scans/overlays/', blank=True, null=True)
    
    # Clinical/Doctor validation input
    doctor_notes = models.TextField(blank=True, null=True)
    final_diagnosis = models.CharField(max_length=100, blank=True, null=True)
    final_severity_index = models.IntegerField(blank=True, null=True)
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='PENDING')

    def __str__(self):
        return f"Scan for {self.patient.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
