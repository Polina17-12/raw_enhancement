const fileInput = document.getElementById('file-upload');
const fileName = document.getElementById('file-name');
const processBtn = document.getElementById('process-btn');
const loader = document.getElementById('loader');
const resultContainer = document.getElementById('result-container');
const resultImg = document.getElementById('result-img');
const placeholder = document.getElementById('placeholder');
const filmstrip = document.getElementById('filmstrip-items');

// Связываем ползунки с текстом значений
const sliders = ['strength', 'exposure', 'temp'];
sliders.forEach(id => {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(`${id}-val`);
    slider.addEventListener('input', (e) => {
        valSpan.innerText = e.target.value + (id === 'strength' ? '%' : '');
    });
});

// Функция загрузки истории из БД
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        if (response.ok) {
            const data = await response.json();
            filmstrip.innerHTML = ''; // Очищаем ленту
            data.forEach(item => {
                appendHistoryCard(item);
            });
        }
    } catch (err) {
        console.error("Ошибка загрузки истории:", err);
    }
}

// Рендеринг карточки в ленту истории
function appendHistoryCard(item) {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
        <img src="${item.output_img}" alt="Preview">
        <div class="info" title="${item.filename}">${item.filename}</div>
    `;
    
    // КЛЮЧЕВАЯ ФИЧА: При клике на историю восстанавливаем параметры и картинку!
    card.addEventListener('click', () => {
        placeholder.style.display = 'none';
        loader.style.display = 'none';
        resultContainer.style.display = 'block';
        resultImg.src = item.output_img;
        
        // Переставляем ползунки
        document.getElementById('strength').value = item.strength;
        document.getElementById('strength-val').innerText = item.strength + '%';
        
        document.getElementById('exposure').value = item.exposure;
        document.getElementById('expo-val').innerText = item.exposure;
        
        document.getElementById('temp').value = item.temp;
        document.getElementById('temp-val').innerText = item.temp;
    });
    
    filmstrip.appendChild(card);
}

// Отслеживание выбора файла
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileName.innerText = e.target.files[0].name;
        fileName.style.color = '#007acc';
    }
});

// Обработка кнопки
processBtn.addEventListener('click', async () => {
    if (fileInput.files.length === 0) {
        alert('Пожалуйста, выберите RAW файл!');
        return;
    }

    placeholder.style.display = 'none';
    resultContainer.style.display = 'none';
    loader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('strength', document.getElementById('strength').value);
    formData.append('exposure', document.getElementById('exposure').value);
    formData.append('temp', document.getElementById('temp').value);

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            resultImg.src = URL.createObjectURL(blob);
            
            loader.style.display = 'none';
            resultContainer.style.display = 'block';
            
            // После успешной генерации обновляем ленту истории
            loadHistory();
        } else {
            alert('Ошибка нейросети.');
            loader.style.display = 'none';
            placeholder.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        loader.style.display = 'none';
        placeholder.style.display = 'block';
    }
});

// Загружаем историю при старте страницы
window.addEventListener('DOMContentLoaded', loadHistory);