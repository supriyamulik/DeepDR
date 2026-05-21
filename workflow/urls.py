from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_scan, name='upload_scan'),
    path('results/<int:scan_id>/', views.scan_results, name='scan_results'),
    path('results/<int:scan_id>/pdf/', views.generate_report_pdf, name='generate_report_pdf'),
    path('grading/', views.grading_queue, name='grading_queue'),
    path('referrals/', views.referrals_queue, name='referrals_queue'),
]
