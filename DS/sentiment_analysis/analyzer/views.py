from django.shortcuts import render,HttpResponse

# Create your views here.
def analyze(request):
     return render(request, 'analyzer.html')
