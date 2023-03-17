
const att = document.createAttribute("style");
att.value='opacity:0.3' 
setInterval(function(){
    $("textarea").keyup(function (){
        if($('textarea').val() !=''){ 
           document.getElementById('one').classList.remove('none')
           document.getElementById('one').classList.add('show')  
        }else{
            document.getElementById('one').classList.add('none');
            document.getElementById('one').classList.remove('show')  
        }
    })
},1000)

$('.button_top').click(function(){
   document.getElementsByClassName('text')[0].setAttribute('disabled', true);
   document.querySelectorAll('div.search')[0].setAttribute("style", "opacity:0.3");
   document.getElementsByClassName('waiting')[0].setAttribute("style", "opacity:1");
})









