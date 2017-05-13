from django.conf.urls import url
from . import views

app_name = "hello_world"

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^[Cc]oncepts?/?$', views.concepts.as_view(), name="Concepts"),
    url(r'^[Ff]inance/?$', views.concepts.as_view(), name="Finance"),
]
