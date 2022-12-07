function show_alarm(type) {
    var style_1_div = 'width: 100%; height: 100%;';
        style_1_div += 'top: 0; left: 0;';
        style_1_div += 'display: block;';
        style_1_div += 'opacity: .8;';
        if(type == 'success'){
            style_1_div += 'background-color: #00ff00;';
        } else if(type == 'warning') {
            style_1_div += 'background-color: #ffff00;';
        } else if(type == 'danger') {
            style_1_div += 'background-color: #ff0000;';
        } else {
            style_1_div += 'background-color: #0000ff;';
        }
        style_1_div += 'position: absolute;';
        style_1_div += 'z-index: 9998;';

    var style_2_div = 'width: 50%; height: 50%;';
        style_2_div += 'top: 0; left: 0;';
        style_2_div += 'display: block;';
        if(type == 'success'){
            style_2_div += 'background-color: #00ff00;';
        } else if(type == 'warning') {
            style_2_div += 'background-color: #ffff00;';
        } else if(type == 'danger') {
            style_2_div += 'background-color: #ff0000;';
        } else {
            style_2_div += 'background-color: #0000ff;';
        }
        style_2_div += 'position: absolute;';
        style_2_div += 'z-index: 9998;';

    var html_1 = '';
        html_1 += '<div class="alarm_background" style="'+style_1_div+'"></div>';
    var html_2 = '';
        html_2 += '<div class="alarm_box" style="'+style_2_div+'"></div>';
    $('body').append(html);
}

function hide_alarm() {
    $('.alarm_background').empty();
    $('.alarm_box').empty();
}