import numpy as np
import tkinter as tk
from tkinter import messagebox

# Ağırlıkları ve Bias değerlerini yükle
W1 = np.loadtxt('W1.txt', delimiter=',')
b1 = np.loadtxt('b1.txt', delimiter=',')
W2 = np.loadtxt('W2.txt', delimiter=',')
b2 = np.loadtxt('b2.txt', delimiter=',')
W3 = np.loadtxt('W3.txt', delimiter=',')
b3 = np.loadtxt('b3.txt', delimiter=',')

class DrawApp:
    def __init__(self, master):
        self.master = master
        master.title("Rakam Çizim Uygulaması")

        self.grid_size = 28
        self.cell_size = 20  # Her hücrenin boyutu
        self.canvas = tk.Canvas(master, width=self.cell_size * self.grid_size, height=self.cell_size * self.grid_size, bg='white')
        self.canvas.pack()

        self.button = tk.Button(master, text="Tahmin Et", command=self.predict)
        self.button.pack()

        self.clear_button = tk.Button(master, text="Temizle", command=self.clear_canvas)
        self.clear_button.pack()

        self.cells = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # Gri tonları için float kullan
        self.is_drawing = False  # Çizim durumunu takip et

        # Fare olayları
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.is_drawing = True
        self.paint(event)

    def paint(self, event):
        if self.is_drawing:
            x = event.x // self.cell_size
            y = event.y // self.cell_size

            # Etki alanı boyutu (daha ince çizim için azaltıldı)
            radius = 1  # Etki alanı çapı (piksellerin etrafında)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                        # Yalnızca etki alanındaki hücreleri koyulaştır
                        self.cells[y + dy, x + dx] = min(1.0, self.cells[y + dy, x + dx] + 0.1)  # Koyulaştırma miktarı azaldı

            self.update_canvas()

    def stop_drawing(self, event):
        self.is_drawing = False

    def draw_cell(self, x, y):
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        color_value = int(self.cells[y, x] * 255)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=f'#{color_value:02x}{color_value:02x}{color_value:02x}', outline='')

    def update_canvas(self):
        self.canvas.delete("all")  # Eski çizimleri temizle
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.draw_cell(x, y)  # Her hücreyi tekrar çiz

    def clear_canvas(self):
        self.canvas.delete("all")
        self.cells.fill(0)  # Tüm hücreleri temizle

    def predict(self):
        # Resmi normalize et
        img = self.cells * 255  # 0-255 aralığına getir
        img = img.reshape(1, -1)  # (1, 784)

        # İleri yayılım
        z1 = np.dot(img, W1) + b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.relu(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = self.softmax(z3)

        # Tahmin
        prediction = np.argmax(a3, axis=1)
        print(f"Tahmin Edilen Rakam: {prediction[0]}")
        messagebox.showinfo("Tahmin", f"Tahmin Edilen Rakam: {prediction[0]}")

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        a = np.exp(x - np.max(x, axis=1, keepdims=True))
        return a / np.sum(a, axis=1, keepdims=True)

# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
