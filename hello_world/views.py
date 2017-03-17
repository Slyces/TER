# from django.shortcuts import get_object_or_404, render
# from django.http import HttpResponseRedirect
# from django.urls import reverse
from django.views import generic
from hello_world.models import Messages


class IndexView(generic.ListView):
    template_name = 'hello_world/index.html'

    def get_queryset(self, *args, **kwargs):
        pass
