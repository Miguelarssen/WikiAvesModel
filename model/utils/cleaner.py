import os
import sys

def clear_augmented(class_name, dataset_root='dataset'):
    """
    Remove todos os arquivos que contenham '_augmented' no nome dentro da classe especificada.
    """
    # Tenta encontrar o dataset partindo da raiz ou do caminho relativo fornecido pelo usuário anteriormente
    possible_paths = [dataset_root, '../../dataset', 'dataset']
    class_path = None
    
    for path in possible_paths:
        tmp_path = os.path.join(path, class_name)
        if os.path.exists(tmp_path):
            class_path = tmp_path
            dataset_root = path
            break
            
    if not class_path:
        print(f"[ERRO] Classe '{class_name}' não encontrada nos caminhos verificados.")
        return

    print(f"[CLEAN] Limpando arquivos aumentados na classe: {class_name} ({class_path})")
    
    # Listar arquivos que contêm '_augmented'
    augmented_files = [f for f in os.listdir(class_path) if f.endswith('.npy') and '_augmented' in f]
    
    if not augmented_files:
        print(f"[INFO] Nenhum arquivo aumentado encontrado para a classe '{class_name}'.")
        return

    count = 0
    for file_name in augmented_files:
        file_path = os.path.join(class_path, file_name)
        try:
            os.remove(file_path)
            count += 1
        except Exception as e:
            print(f"[ERRO] Falha ao deletar {file_name}: {e}")

    print(f"[OK] {count} arquivos aumentados foram removidos com sucesso.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clear_augmented(sys.argv[1])
    else:
        print("Uso: python cleaner.py <nome_da_classe>")
