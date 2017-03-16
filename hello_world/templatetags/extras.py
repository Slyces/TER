from django import template

register = template.Library()


@register.filter
def modulo(num, val):
    return num % val


@register.filter
def align(user):
    return {'Petite Marmotte': 'right', 'Petite Pétale de Rose': 'left'}[user]


@register.filter
def sameblock(array, index):
    return index != 0 or array[index].block == array[index+1].block


@register.filter
def last_info(array, index):
    index = 1 if index == 0 else index
    return str(array[index-1]) + '-' + str(array[index-1].origin)