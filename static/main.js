const fileInput = document.getElementById('file-upload');
const fileName = document.getElementById('file-name');
const processBtn = document.getElementById('process-btn');
const loader = document.getElementById('loader');
const resultContainer = document.getElementById('result-container');
const placeholder = document.getElementById('placeholder');
const filmstrip = document.getElementById('filmstrip-items');

const comparisonSlider = document.getElementById('comparison-slider');
const sliderOverlay = document.getElementById('slider-overlay');
const sliderLine = document.getElementById('slider-line');
const beforeImg = document.getElementById('before-img');
const afterImg = document.getElementById('after-img');

// Логика движения ползунка
comparisonSlider.addEventListener('input', (e) => {
    const val = e.target.value + '%';
    sliderOverlay.style.width = val;
    sliderLine.style.left = val;
});

// Синхронизация размеров скрытой картинки с оригиналом
function syncSizes() {
    if (beforeImg.complete) {
        afterImg.style.width = beforeImg.offsetWidth + 'px';
        afterImg.style.height = beforeImg.offsetHeight + 'px';
    }
}
beforeImg.addEventListener('load', syncSizes);
window.addEventListener('resize', syncSizes);

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
            filmstrip.innerHTML = ''; 
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
    
    // При клике на историю восстанавливаем параметры и картинку
    card.addEventListener('click', () => {
        placeholder.style.display = 'none';
        loader.style.display = 'none';
        resultContainer.style.display = 'block';
        
        // Вычисляем путь к оригиналу на лету
        const origPath = item.output_img.replace('result_', 'orig_');
        beforeImg.src = origPath;
        afterImg.src = item.output_img;
        
        // Центрируем ползунок при переключении истории
        comparisonSlider.value = 50;
        sliderOverlay.style.width = '50%';
        sliderLine.style.left = '50%';
        setTimeout(syncSizes, 50);
        
        // Переставляем ползунки
        document.getElementById('strength').value = item.strength;
        document.getElementById('strength-val').innerText = item.strength + '%';
        document.getElementById('exposure').value = item.exposure;
        document.getElementById('exposure-val').innerText = item.exposure;
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

// Обработка кнопки запуска
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
            const data = await response.json(); 
            
            beforeImg.src = data.original;
            afterImg.src = data.processed;
            
            // Сбрасываем ползунок по центру
            comparisonSlider.value = 50;
            sliderOverlay.style.width = '50%';
            sliderLine.style.left = '50%';

            loader.style.display = 'none';
            resultContainer.style.display = 'block';
            setTimeout(syncSizes, 50);
            
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