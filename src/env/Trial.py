from multiagent_env import MulticoffeEnv
from paralel_env import MulticoffeParallelEnv
from pettingzoo.test import parallel_api_test
        
if __name__ == '__main__':
    import pandas as pd
    
    df = pd.read_csv('df_train.csv')

    # Seleccionar qué entorno probar (AEC o Paralelo)
    env_type = "parallel"  # Cambiar a "aec" para probar el otro entorno
    
    if env_type == "parallel":
        print("\n=== Probando entorno PARALELO ===")
        env = MulticoffeParallelEnv(df=df, render_mode="human")
        
        # Prueba de API para entornos paralelos
        print("\nEjecutando prueba de API Paralela de PettingZoo...")
        parallel_api_test(env, num_cycles=1000)
        print("Prueba de API Paralela completada.")
        
    else:
        print("\n=== Probando entorno AEC ===")
        env = MulticoffeEnv(df=df, render_mode="human")
        
        # Prueba de API para entornos AEC
        from pettingzoo.test import api_test
        print("\nEjecutando prueba de API AEC de PettingZoo...")
        api_test(env, num_cycles=1000)
        print("Prueba de API AEC completada.")

    # Ejecución manual de episodios (adaptado al tipo de entorno)
    num_episodes = 3  # Número de episodios a ejecutar

    for episode in range(num_episodes):
        print(f"\n--- Iniciando Episodio {episode + 1} ---")
        
        if env_type == "parallel":
            # Ejecución para entorno paralelo
            observations = env.reset()
            env.render()  # Render inicial del episodio
            
            # Todos los agentes actúan simultáneamente
            actions = {
                "Mix": env.action_space("Mix").sample(),
                "Additive": env.action_space("Additive").sample(),
                "Container": env.action_space("Container").sample()
        }
            print("\nAcciones a tomar:")
            for agent, action in actions.items():
                print(f"{agent}: {action}")
            
            # Un solo paso para todos los agentes
            observations, rewards, terminations, truncations, infos = env.step(actions)
            env.render()
            
            print(f"\nRecompensas: {rewards}")
            print(f"Terminaciones: {terminations}")
            print(f"Infos: {infos}")
            
            print(f"\n--- EPISODIO {episode + 1} TERMINADO ---")
            
        else:
            # Ejecución para entorno AEC (turnos secuenciales)
            env.reset()
            env.render()  # Render inicial del episodio
            
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                    # Solo mostrar mensaje de terminación una vez al final
                    if all(env.terminations.values()) or all(env.truncations.values()):
                        print(f"\n--- EPISODIO {episode + 1} TERMINADO ---")
                        break
                else:
                    action = env.action_space(agent).sample()
                
                env.step(action)
                
    env.close()