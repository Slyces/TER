from django import template

register = template.Library()


@register.filter
def key(dictionnary, key_name):
    print("dict keys :", dictionnary.keys())
    print("key :", key_name)
    try:
        value = dictionnary[key_name]
    except KeyError:
        from django.conf import settings
        value = settings.TEMPLATE_STRING_IF_INVALID

    return value

@register.filter
def active(key):
    return "active" if key == 'Rates' else ""
