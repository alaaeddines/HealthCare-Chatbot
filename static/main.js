            const popup = document.querySelector('.chat-popup');
            const chatBtn = document.querySelector('.chat-btn');
            const submitBtn = document.querySelector('.submit');
            const chatArea = document.querySelector('.chat-area');
            const inputElm = document.querySelector('input');
            const chatUrl = '/chatbot';


            // chat button toggler

            chatBtn.addEventListener('click', ()=>{
                popup.classList.toggle('show');
            })

            // send msg
            submitBtn.addEventListener('click', ()=>{
                let userInput = inputElm.value;
                
                if (userInput == ''){}

                else{
                    let temp = `<div class = "out-msg">
                    <span class = "my-msg">${userInput}</span>
                    </div>`;
                    chatArea.insertAdjacentHTML("beforeend", temp);
                    inputElm.value = '';
                }
                fetch(chatUrl, {
                    method:"POST", 
                    body: JSON.stringify({"msg":userInput}), 
                    headers:{"Content-type":"application/json; charset=UTF-8"}}
                ).then(res => res.json()).then(function(data){
                    let temp2 = `<div class="income-msg">
                        <span class="msg">${data.msg}</span>
                        </div>`;
                    chatArea.insertAdjacentHTML("beforeend", temp2);
                    // console.log(data);
                    scrollToBottom();
                    });
            });
            function scrollToBottom() {
                messages.scrollTop = messages.scrollHeight;
              }